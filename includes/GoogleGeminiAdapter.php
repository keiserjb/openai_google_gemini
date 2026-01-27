<?php

/**
 * @file
 * Google Gemini adapter for the OpenAI module.
 *
 * This adapter implements AIClientInterface using Google's Gemini REST API.
 * No external SDK dependency - uses Backdrop's built-in HTTP functions.
 *
 * @see https://ai.google.dev/docs
 */


class GoogleGeminiAdapter implements AIClientInterface {

  /** @var string */
  protected $apiKey;

  /** @var string */
  protected $baseUrl = 'https://generativelanguage.googleapis.com/v1beta/openai/';

  /**
   * Constructor.
   *
   * @param string $apiKey
   *   Google Gemini API key.
   */
  public function __construct($apiKey) {
    $this->apiKey = trim($apiKey);

    if (empty($this->apiKey)) {
      throw new \Exception('Google Gemini API key is required');
    }
  }

  /**
   * {@inheritdoc}
   */
  public function getModels(): array {
    // Use the correct Gemini endpoint for model listing, not the OpenAI compatibility path.
    try {
      $url = 'https://generativelanguage.googleapis.com/v1/models?key=' . urlencode($this->apiKey);
      $options = [
        'method' => 'GET',
        'headers' => [
          'Accept' => 'application/json',
        ],
        'timeout' => 10,
      ];

      $response = backdrop_http_request($url, $options);
      if (!isset($response->code) || (int) $response->code !== 200) {
        return [];
      }

      $data = json_decode($response->data, TRUE);
      $models = [];

      if (!empty($data['models']) && is_array($data['models'])) {
        foreach ($data['models'] as $item) {
          $name = $item['name'] ?? null;
          $display = $item['displayName'] ?? $item['description'] ?? $name;
          if (!$name) continue;
          $models[$name] = $name . ' — ' . $display;
        }
      }

      if (!empty($models)) {
        asort($models);
        return $models;
      }
    }
    catch (\Exception $e) {
      watchdog('openai_google_gemini', 'Failed to fetch Gemini models: @error', ['@error' => $e->getMessage()], WATCHDOG_WARNING);
    }
    return [];
  }

  /**
   * {@inheritdoc}
   */
  public function completions(string $model, string $prompt, $temperature, $max_tokens = 512, bool $stream_response = FALSE) {
    try {
      // Google Gemini doesn't have a separate "completions" endpoint like OpenAI
      // We use the chat/messages API with a single user message
      $messages = [
        [
          'role' => 'user',
          'content' => trim($prompt),
        ],
      ];

      return $this->chat($model, $messages, $temperature, $max_tokens, $stream_response);
    }
    catch (\Exception $e) {
      watchdog('openai_google_gemini', 'Completions error: @error',
        ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * Get models by their capability.
   */
  public function getModelsByCapability($capability): array {
    $models = $this->getModels();
    if ($capability === 'text') {
      return $models;
    }
    if ($capability === 'vision') {
      $vision_models = [];
      foreach ($models as $id => $label) {
        // Gemini 1.5 and 2.0 models are multimodal (text/vision)
        if (preg_match('/gemini-(1\.5|2\.0)/i', $id)) {
          $vision_models[$id] = $label;
        }
      }
      return $vision_models;
    }
    return [];
  }

  /**
   * {@inheritdoc}
   */
  public function chat(string $model, array $messages, $temperature, $max_tokens = 1024, bool $stream_response = FALSE) {
    try {
      // Convert messages to Gemini-native format
      $gemini_contents = $this->_convertMessagesGemini($messages);
      $body = [
        'contents' => $gemini_contents,
        'generationConfig' => [
          'maxOutputTokens' => (int) $max_tokens ?: 1024,
          'temperature' => (float) $temperature,
        ],
      ];
      // Normalize the incoming model identifier into the Gemini resource
      // format the API expects (e.g. "models/gemini-1.5-pro"). The model
      // string may come in several shapes depending on the higher-level UI
      // (e.g. "google_gemini/models/gemini-1.5-pro", "models/gemini-1.5-pro",
      // or just "gemini-1.5-pro"). Normalize to a single "models/..." form
      // and then append to the endpoint path.
      $resource = $model;
      // If the string contains 'models/' anywhere, take that segment onward.
      if (strpos($resource, 'models/') !== FALSE) {
        $resource = substr($resource, strpos($resource, 'models/'));
      }
      elseif (strpos($resource, '/') !== FALSE) {
        // If other prefix exists (e.g., 'provider/model'), take the last segment.
        $resource = substr($resource, strrpos($resource, '/') + 1);
        $resource = 'models/' . $resource;
      }
      else {
        // Bare id (e.g., 'gemini-1.5-pro') -> prefix with models/.
        $resource = 'models/' . $resource;
      }

      $url = 'https://generativelanguage.googleapis.com/v1beta/' . $resource . ':generateContent?key=' . urlencode($this->apiKey);
      if ($stream_response) {
        // Gemini streaming not implemented in this adapter yet
        throw new \Exception('Streaming not implemented for Gemini-native API');
      }

      // Primary attempt: call the per-model generateContent endpoint.
      // Log the normalized resource and body keys for debugging.
      watchdog('openai_google_gemini', 'Gemini generateContent primary attempt for @resource with keys: @keys', [
        '@resource' => $resource,
        '@keys' => implode(',', array_keys($body)),
      ], WATCHDOG_DEBUG);

      try {
        $response = $this->_makeRequest($url, $body);
      }
      catch (\Exception $e) {
        $msg = $e->getMessage();
        // Detect a broader set of model/payload errors returned by Gemini.
        if (preg_match('/(unexpected model name format|GenerateContentRequest\.model|Invalid JSON payload|Unknown name)/i', $msg)) {
          watchdog('openai_google_gemini', 'Model format/payload error for @resource: @msg; retrying with alternate endpoint', ['@resource' => $resource, '@msg' => $msg], WATCHDOG_WARNING);

          $alt_url = 'https://generativelanguage.googleapis.com/v1beta/models:generateContent?key=' . urlencode($this->apiKey);
          // Include explicit model field using the canonical resource.
          $body_with_model = $body;
          $body_with_model['model'] = $resource;

          try {
            watchdog('openai_google_gemini', 'Retrying Gemini generateContent via models:generateContent with model=@m', ['@m' => $resource], WATCHDOG_DEBUG);
            $response = $this->_makeRequest($alt_url, $body_with_model);
          }
          catch (\Exception $e2) {
            // As a last-ditch attempt, try the bare model id (without 'models/').
            $bare = preg_replace('#^models/#i', '', $resource);
            $body_with_model['model'] = $bare;
            try {
              watchdog('openai_google_gemini', 'Retrying Gemini generateContent with bare model=@m', ['@m' => $bare], WATCHDOG_DEBUG);
              $response = $this->_makeRequest($alt_url, $body_with_model);
            }
            catch (\Exception $e3) {
              // Log all three failures and rethrow the last exception.
              watchdog('openai_google_gemini', 'All Gemini generateContent attempts failed for @resource. Errors: primary=@err1 alt=@err2 last=@err3', ['@resource' => $resource, '@err1' => $msg, '@err2' => $e2->getMessage(), '@err3' => $e3->getMessage()], WATCHDOG_ERROR);
              throw $e3;
            }
          }
        }
        else {
          throw $e;
        }
      }

      // Extract text from Gemini response
      $text = $this->_extractCandidateText($response);
      return trim($text);
    }
    catch (\Exception $e) {
      watchdog('openai_google_gemini', 'Chat error: @error',
        ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * Extract the primary candidate text from a Gemini response.
   *
   * Handles several response shapes safely and concatenates multiple parts
   * when present.
   *
   * @param array|null $response
   *   Decoded response array from Gemini API.
   *
   * @return string
   *   The extracted text or empty string when none found.
   */
  protected function _extractCandidateText($response): string {
    if (empty($response) || !is_array($response)) {
      return '';
    }

    // Look for candidates -> content -> parts -> text
    if (!empty($response['candidates']) && is_array($response['candidates'])) {
      $first = $response['candidates'][0];
      if (!empty($first['content'])) {
        $content = $first['content'];
        // If content.parts is present, join any 'text' entries.
        if (!empty($content['parts']) && is_array($content['parts'])) {
          $pieces = [];
          foreach ($content['parts'] as $part) {
            if (is_string($part) && trim($part) !== '') {
              $pieces[] = $part;
            }
            elseif (is_array($part) && isset($part['text'])) {
              $pieces[] = $part['text'];
            }
          }
          if (!empty($pieces)) {
            return implode("\n", $pieces);
          }
        }
        // Some variants may put text directly under content
        if (isset($content['text']) && is_string($content['text'])) {
          return $content['text'];
        }
      }
      // Fallback: sometimes candidate may contain a top-level 'text' key
      if (isset($first['text']) && is_string($first['text'])) {
        return $first['text'];
      }
    }

    // Fallback: check for other common shapes such as 'output' or 'message'.
    if (!empty($response['output']) && is_string($response['output'])) {
      return $response['output'];
    }
    if (!empty($response['message']) && is_string($response['message'])) {
      return $response['message'];
    }

    return '';
  }

  /**
   * {@inheritdoc}
   */
  public function images(string $model, string $prompt, string $size, string $response_format, string $quality = 'standard', string $style = 'natural', ?string $output_format = NULL) {
    watchdog('openai_google_gemini',
      'Image generation API endpoint not yet implemented for Gemini. Use OpenAI for image generation.',
      [], WATCHDOG_WARNING);

    return [
      'data' => [],
      'error' => 'Image generation is not yet fully implemented in Gemini adapter. Use OpenAI or OpenRouter instead.',
    ];
  }

  /**
   * {@inheritdoc}
   */
  public function textToSpeech(string $model, string $input, string $voice, string $response_format) {
    watchdog('openai_google_gemini',
      'Text-to-speech is not supported by Google Gemini adapter.',
      [], WATCHDOG_WARNING);

    return '';
  }

  /**
   * {@inheritdoc}
   */
  public function speechToText(string $model, string $file, string $task = 'transcribe', $temperature = 0.4, string $response_format = 'verbose_json') {
    watchdog('openai_google_gemini',
      'Speech-to-text is not supported by Google Gemini adapter.',
      [], WATCHDOG_WARNING);

    return [];
  }

  /**
   * {@inheritdoc}
   */
  public function moderation(string $input, string $model = 'gemini-moderation'): array {
    watchdog('openai_google_gemini',
      'Moderation API is not directly available in Gemini. Gemini has built-in safety filters.',
      [], WATCHDOG_WARNING);

    return [
      'results' => [],
      'error' => 'Moderation endpoint is not directly available in Google Gemini adapter.',
    ];
  }

  /**
   * {@inheritdoc}
   *
   * Backwards-compatible single-input embedding wrapper that calls the bulk
   * embeddings() helper and returns the first embedding result in the same
   * predictable shape as other adapters (object/data/index).
   */
  public function embedding(string $input, string $model, bool $log = TRUE): array {
    $start_time = microtime(TRUE);
    try {
      // Reuse the bulk embeddings method with a single-item array.
      $result = $this->embeddings($model, [$input]);
      if (!empty($result['data']) && is_array($result['data']) && isset($result['data'][0])) {
        if (isset($this->api) && method_exists($this->api, 'recordLog')) {
          $duration = microtime(TRUE) - $start_time;
          $this->api->recordLog('embedding', $model, ['input' => $input], $result, TRUE, $duration, NULL, !$log);
        }
        // Return only the vector to match the new AIClientInterface::embedding contract
        return is_array($result['data'][0]) ? ($result['data'][0]['embedding'] ?? []) : [];
      }
      return [];
    }
    catch (\Exception $e) {
      if (isset($this->api) && method_exists($this->api, 'recordLog')) {
        $duration = microtime(TRUE) - $start_time;
        $this->api->recordLog('embedding', $model, ['input' => $input], NULL, FALSE, $duration, $e->getMessage(), !$log);
      }
      if ($log) {
        $error_msg = $e->getMessage();
        // Suppress log if it's a "does not support embeddings" or similar during probing.
        if (strpos($error_msg, 'does not support embeddings') === FALSE && strpos($error_msg, 'not found') === FALSE) {
          watchdog('openai_google_gemini', 'Embedding error: @error', ['@error' => $error_msg], WATCHDOG_ERROR);
        }
      }
      return [];
    }
  }

  /**
   * {@inheritdoc}
   */
  public function embeddings(string $model, array $inputs, string $response_format = 'float'): array {
    try {
      // Google Gemini supports embeddings via the embedding API
      $embedding_model = 'embedding-001';
      $results = [];

      $url = $this->baseUrl . 'embeddings?key=' . urlencode($this->apiKey);

      foreach ($inputs as $index => $input) {
        $body = [
          'model' => $embedding_model,
          'content' => [
            'parts' => [
              [
                'text' => $input,
              ],
            ],
          ],
        ];

        $response = $this->_makeRequest($url, $body);

        if (isset($response['embedding']['values'])) {
          $results[] = [
            'object' => 'embedding',
            'embedding' => $response['embedding']['values'],
            'index' => $index,
          ];
        }
      }

      return [
        'object' => 'list',
        'data' => $results,
        'model' => $embedding_model,
        'usage' => [
          'prompt_tokens' => count($inputs),
          'total_tokens' => count($inputs),
        ],
      ];
    }
    catch (\Exception $e) {
      watchdog('openai_google_gemini', 'Embeddings error: @error',
        ['@error' => $e->getMessage()], WATCHDOG_ERROR);

      return [
        'data' => [],
        'error' => $e->getMessage(),
      ];
    }
  }

  /**
   * Convert OpenAI-style messages to Gemini-native format.
   *
   * @param array $messages
   *   Array of messages in OpenAI format.
   *
   * @return array
   *   Converted messages for Gemini API.
   */
  protected function _convertMessagesGemini(array $messages): array {
    $gemini_contents = [];
    foreach ($messages as $msg) {
      $role = $msg['role'] ?? 'user';
      $gemini_role = ($role === 'assistant') ? 'model' : 'user';
      if ($role === 'system') {
        continue;
      }
      $gemini_contents[] = [
        'role' => $gemini_role,
        'parts' => [
          [ 'text' => $msg['content'] ],
        ],
      ];
    }
    return $gemini_contents;
  }

  /**
   * Make HTTP request to Gemini API.
   *
   * @param string $url
   *   The API endpoint URL.
   * @param array $body
   *   The request body.
   *
   * @return array
   *   The parsed JSON response.
   */
  protected function _makeRequest($url, array $body) {
    $options = [
      'method' => 'POST',
      'headers' => [
        'Content-Type' => 'application/json',
      ],
      'data' => json_encode($body),
      'timeout' => 30,
    ];

    $response = backdrop_http_request($url, $options);

    $code = isset($response->code) ? (int) $response->code : 0;
    $body_text = isset($response->data) ? $response->data : '';

    // Mask API key in URLs for logging (- keep only hostname/path)
    $masked_url = preg_replace('/(key=)[^&\s]+/i', '$1***', $url);
    $log_body_keys = is_array($body) ? implode(',', array_keys($body)) : '';

    // On non-2xx responses, log a short, masked diagnostic to watchdog for debugging.
    if ($code < 200 || $code >= 300) {
      $trim_resp = is_string($body_text) ? substr($body_text, 0, 1000) : json_encode($body_text);
      watchdog('openai_google_gemini', 'Gemini request failed. url=@url; code=@code; body_keys=@keys; resp_snippet=@resp', [
        '@url' => $masked_url,
        '@code' => $code,
        '@keys' => $log_body_keys,
        '@resp' => $trim_resp,
      ], WATCHDOG_WARNING);
    }

    // Treat any 2xx as success.
    if ($code >= 200 && $code < 300) {
      return json_decode($body_text, TRUE);
    }

    // For non-2xx responses, attempt to decode the body. Some Gemini
    // responses can include a success-shaped payload even when the status
    // is unexpected; if so, treat them as success to avoid false errors.
    $decoded = NULL;
    // If the returned data is already an array/object (some wrappers do
    // this), normalize it to an array.
    if (is_array($body_text)) {
      $decoded = $body_text;
    }
    elseif (is_object($body_text)) {
      $decoded = (array) $body_text;
    }
    else {
      // Attempt normal JSON decode first.
      $decoded = json_decode((string) $body_text, TRUE);

      // If decode failed but the raw text contains recognizable keys,
      // try extracting the JSON object substring and decode that.
      if ($decoded === NULL) {
        $raw = (string) $body_text;
        if (stripos($raw, 'candidates') !== FALSE || stripos($raw, 'modelVersion') !== FALSE) {
          // Find a balanced JSON object around the anchor (e.g., "candidates").
          $anchor_pos = stripos($raw, 'candidates');
          if ($anchor_pos === FALSE) {
            $anchor_pos = stripos($raw, 'modelVersion');
          }
          if ($anchor_pos !== FALSE) {
            // Find the last '{' before the anchor.
            $start = null;
            for ($i = $anchor_pos; $i >= 0; $i--) {
              if ($raw[$i] === '{') { $start = $i; break; }
            }
            if ($start !== null) {
              $depth = 0;
              $end = null;
              $len = strlen($raw);
              for ($j = $start; $j < $len; $j++) {
                if ($raw[$j] === '{') { $depth++; }
                elseif ($raw[$j] === '}') { $depth--; }
                if ($depth === 0) { $end = $j; break; }
              }
              if ($end !== null && $end > $start) {
                $try = substr($raw, $start, $end - $start + 1);
                $decoded = json_decode($try, TRUE);
              }
            }
          }
         }
       }
     }

     if (is_array($decoded) && (!empty($decoded['candidates']) || !empty($decoded['modelVersion']) || !empty($decoded['usageMetadata']))) {
      watchdog('openai_google_gemini', 'Gemini returned non-2xx but success-looking payload (code=@code): @keys', ['@code' => $code, '@keys' => implode(',', array_keys($decoded))], WATCHDOG_WARNING);
      return $decoded;
    }

    // As a last resort: if the raw body contains candidate-like text but we
    // couldn't parse it, synthesize a minimal response so higher-level code
    // can extract the text instead of failing completely. This avoids
    // crashing when providers return slightly malformed/extra-wrapped JSON.
    $raw = is_string($body_text) ? $body_text : (is_scalar($body_text) ? (string) $body_text : json_encode($body_text));
    if (stripos($raw, 'candidates') !== FALSE || stripos($raw, 'modelVersion') !== FALSE || stripos($raw, 'usageMetadata') !== FALSE) {
      watchdog('openai_google_gemini', 'Gemini returned candidate-like payload but could not decode JSON; returning synthetic candidate.', [], WATCHDOG_WARNING);
      return [
        'candidates' => [
          [
            'content' => [
              'parts' => [
                [ 'text' => $raw ],
              ],
            ],
          ],
        ],
      ];
    }

    $error = is_string($body_text) ? $body_text : json_encode($body_text);
    throw new \Exception("Gemini API error: " . $error);
  }

  /**
   * Handle streaming responses for Gemini.
   *
   * @param string $url
   *   The API endpoint URL.
   * @param array $body
   *   The request body.
   *
   * @return mixed
   *   A streaming HTTP response.
   */
  protected function _handleStreamingRequest($url, array $body) {
    // Use an anonymous streaming responder (no external StreamedResponse
    // dependency) similar to the Anthropic adapter's approach. The caller
    // can invoke ->send() to emit the streamed data.
    return new class($url, $body) {
      protected $url;
      protected $body;

      public function __construct($url, $body) {
        $this->url = $url;
        $this->body = $body;
      }

      public function send() {
        $options = [
          'method' => 'POST',
          'headers' => [
            'Content-Type' => 'application/json',
          ],
          'data' => json_encode($this->body),
          'timeout' => 300,
          'stream' => TRUE,
        ];

        try {
          $response = backdrop_http_request($this->url, $options);

          if (!isset($response->code) || $response->code !== 200) {
            echo "Error: Gemini API returned " . (isset($response->code) ? $response->code : 'unknown');
            return;
          }

          if (isset($response->data) && is_string($response->data)) {
            $lines = explode("\n", $response->data);
            foreach ($lines as $line) {
              if (strpos($line, 'data: ') === 0) {
                $json = substr($line, 6);
                $data = json_decode($json, TRUE);
                if (isset($data['choices'][0]['delta']['content'])) {
                  echo $data['choices'][0]['delta']['content'];
                  @ob_flush();
                  @flush();
                }
              }
            }
          }
        }
        catch (\Exception $e) {
          echo "Streaming error: " . $e->getMessage();
        }
      }
    };
  }
}
