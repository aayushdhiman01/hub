use crate::config::constants::default_max_tokens;use crate::models::chat::{ChatCompletion, ChatCompletionChoice};
use crate::models::content::{ChatCompletionMessage, ChatMessageContent};
use crate::models::embeddings::{
    Embeddings, EmbeddingsInput, EmbeddingsRequest, EmbeddingsResponse,
};
use crate::models::streaming::{ChatCompletionChunk, Choice, ChoiceDelta};
use crate::models::usage::Usage;
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

impl From<crate::models::chat::ChatCompletionRequest> for BedrockChatCompletionRequest {
    fn from(request: crate::models::chat::ChatCompletionRequest) -> Self {
        BedrockChatCompletionRequest {
            input: request
                .messages
                .iter()
                .map(|message| match &message.content {
                    Some(ChatMessageContent::String(text)) => text.clone(),
                    Some(ChatMessageContent::Array(parts)) => parts
                        .iter()
                        .map(|part| part.text.clone())
                        .collect::<Vec<_>>()
                        .join(" "),
                    _ => String::new(),
                })
                .collect::<Vec<String>>()
                .join("\n"),  // Assuming the messages are concatenated as a single input string
            model: "bedrock-model".to_string(),  // Replace with actual model name
            max_tokens: request.max_tokens.or(Some(default_max_tokens())),
            temperature: request.temperature,
            top_p: request.top_p,
            n: request.n.map(|n| n as u32),
        }
    }
}

impl From<BedrockChatCompletionResponse> for ChatCompletion {
    fn from(response: BedrockChatCompletionResponse) -> Self {
        let choices = response
            .choices
            .into_iter()
            .enumerate()
            .map(|(index, choice)| ChatCompletionChoice {
                index: index as u32,
                message: ChatCompletionMessage {
                    role: choice.message.role.clone(),
                    content: Some(ChatMessageContent::String(choice.message.content)),
                    name: None,
                    tool_calls: None,
                },
                finish_reason: Some(choice.finish_reason),
                logprobs: None,
            })
            .collect();

        ChatCompletion {
            id: uuid::Uuid::new_v4().to_string(),
            object: None,
            created: None,
            model: "bedrock-model".to_string(),  // Replace with actual model name
            choices,
            usage: Usage::default(),
            system_fingerprint: None,
        }
    }
}

impl From<EmbeddingsRequest> for BedrockEmbeddingsRequest {
    fn from(request: EmbeddingsRequest) -> Self {
        BedrockEmbeddingsRequest {
            input: match request.input {
                EmbeddingsInput::Single(text) => vec![text],
                EmbeddingsInput::Multiple(texts) => texts,
            },
            model: "bedrock-embedding-model".to_string(),  // Replace with actual embedding model name
        }
    }
}

impl From<BedrockEmbeddingsResponse> for EmbeddingsResponse {
    fn from(response: BedrockEmbeddingsResponse) -> Self {
        let token_count = response
            .embeddings
            .iter()
            .flat_map(|embedding| embedding.values.iter())
            .count() as u32;

        EmbeddingsResponse {
            object: "list".to_string(),
            data: response
                .embeddings
                .into_iter()
                .enumerate()
                .map(|(index, embedding)| Embeddings {
                    object: "embedding".to_string(),
                    embedding: embedding.values,
                    index,
                })
                .collect(),
            model: "bedrock-embedding-model".to_string(),
            usage: Usage {
                prompt_tokens: token_count,
                completion_tokens: 0,
                total_tokens: token_count,
                completion_tokens_details: None,
                prompt_tokens_details: None,
            },
        }
    }
}