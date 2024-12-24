use serde::{Deserialize, Serialize};

/// Request model for Bedrock Chat Completion
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct BedrockChatCompletionRequest {
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<BedrockGenerationParameters>,
}

/// Generation parameters for chat completions
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct BedrockGenerationParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

/// Response model for Bedrock Chat Completion
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct BedrockChatCompletionResponse {
    pub results: Vec<BedrockResult>,
}

/// Result in the chat completion response
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct BedrockResult {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Request model for Bedrock Embeddings
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct BedrockEmbeddingsRequest {
    pub text: String,
}

/// Response model for Bedrock Embeddings
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct BedrockEmbeddingsResponse {
    pub embedding: Vec<f32>,
}

/// Request model for Bedrock Completions (similar to chat completions, can be reused)
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct BedrockCompletionsRequest {
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<BedrockGenerationParameters>,
}

/// Response model for Bedrock Completions
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct BedrockCompletionsResponse {
    pub results: Vec<BedrockResult>,
}

/// Streaming chunk for Bedrock API response (if applicable for streaming use case)
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct BedrockStreamChunk {
    pub results: Vec<BedrockStreamResult>,
}

/// Streaming result in the chunk
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct BedrockStreamResult {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Metadata for the Bedrock API usage (e.g., token counts)
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct BedrockUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: i32,
    #[serde(rename = "completionTokenCount")]
    pub completion_token_count: i32,
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: i32,
}
