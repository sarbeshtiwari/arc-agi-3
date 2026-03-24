/*
  LLMProvider — unified interface for calling multiple LLM providers.
  Supports AWS Bedrock (Converse API), OpenAI Chat Completions, and Google Gemini.
  Clients are initialised lazily on first use.

  Environment variables:
    AWS_BEDROCK_ACCESS_KEY_ID, AWS_BEDROCK_SECRET_ACCESS_KEY,
    AWS_BEDROCK_SESSION_TOKEN, AWS_BEDROCK_REGION
    OPENAI_API_KEY
    GOOGLE_API_KEY
*/

import {
  BedrockRuntimeClient,
  ConverseCommand,
  type Message as BedrockMessage,
} from '@aws-sdk/client-bedrock-runtime';
import OpenAI from 'openai';
import { GoogleGenAI } from '@google/genai';

// ── Model registry ────────────────────────────────────────────────────────

export interface ModelConfig {
  id: string;
  name: string;
  provider: 'bedrock' | 'openai' | 'gemini';
  modelId: string; // actual API model ID
  color: string; // for UI
  supportsStreaming: boolean;
}

export const AVAILABLE_MODELS: ModelConfig[] = [
  {
    id: 'chatgpt',
    name: 'ChatGPT',
    provider: 'openai',
    modelId: 'gpt-4o',
    color: '#10B981',
    supportsStreaming: true,
  },
  {
    id: 'gemini',
    name: 'Gemini',
    provider: 'gemini',
    modelId: 'gemini-2.0-flash',
    color: '#3B82F6',
    supportsStreaming: true,
  },
  {
    id: 'kimi-k2.5',
    name: 'Kimi k2.5',
    provider: 'bedrock',
    modelId: 'moonshotai.kimi-k2.5',
    color: '#8B5CF6',
    supportsStreaming: true,
  },
  {
    id: 'claude-opus-4.6',
    name: 'Claude Opus 4.6',
    provider: 'bedrock',
    modelId: 'us.anthropic.claude-opus-4-20250514-v1:0',
    color: '#D946EF',
    supportsStreaming: true,
  },
];

// ── Bedrock client (lazy singleton) ───────────────────────────────────────

let _bedrockClient: BedrockRuntimeClient | null = null;

function getBedrockClient(): BedrockRuntimeClient {
  if (!_bedrockClient) {
    const region =
      process.env.AWS_BEDROCK_REGION || process.env.AWS_REGION || 'us-east-1';
    const accessKeyId =
      process.env.AWS_BEDROCK_ACCESS_KEY_ID || process.env.AWS_ACCESS_KEY_ID;
    const secretAccessKey =
      process.env.AWS_BEDROCK_SECRET_ACCESS_KEY ||
      process.env.AWS_SECRET_ACCESS_KEY;

    if (!accessKeyId || !secretAccessKey) {
      throw new Error(
        'AWS Bedrock credentials not configured. ' +
          'Set AWS_BEDROCK_ACCESS_KEY_ID and AWS_BEDROCK_SECRET_ACCESS_KEY.',
      );
    }

    _bedrockClient = new BedrockRuntimeClient({
      region,
      credentials: {
        accessKeyId,
        secretAccessKey,
        sessionToken: process.env.AWS_BEDROCK_SESSION_TOKEN || undefined,
      },
    });
  }
  return _bedrockClient;
}

// ── OpenAI client (lazy singleton) ────────────────────────────────────────

let _openaiClient: OpenAI | null = null;

function getOpenAIClient(): OpenAI {
  if (!_openaiClient) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error(
        'OPENAI_API_KEY not configured. Set it in your .env file.',
      );
    }
    _openaiClient = new OpenAI({ apiKey });
  }
  return _openaiClient;
}

// ── Google Gemini client (lazy singleton) ──────────────────────────────────

let _geminiClient: GoogleGenAI | null = null;

function getGeminiClient(): GoogleGenAI {
  if (!_geminiClient) {
    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) {
      throw new Error(
        'GOOGLE_API_KEY not configured. Set it in your .env file.',
      );
    }
    _geminiClient = new GoogleGenAI({ apiKey });
  }
  return _geminiClient;
}

// ── Types ─────────────────────────────────────────────────────────────────

export interface LLMMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface LLMCallOptions {
  maxTokens?: number;
  temperature?: number;
}

export interface LLMResponse {
  content: string;
  inputTokens: number;
  outputTokens: number;
}

// ── Provider implementations ──────────────────────────────────────────────

/**
 * Call AWS Bedrock via the Converse API (non-streaming).
 * System messages are passed via the `system` parameter; user/assistant
 * messages go into the `messages` array.
 */
async function callBedrock(
  modelConfig: ModelConfig,
  messages: LLMMessage[],
  options?: LLMCallOptions,
): Promise<LLMResponse> {
  const client = getBedrockClient();

  // Separate system messages from conversation messages
  const systemTexts = messages
    .filter((m) => m.role === 'system')
    .map((m) => ({ text: m.content }));

  const conversationMessages: BedrockMessage[] = messages
    .filter((m) => m.role !== 'system')
    .map((m) => ({
      role: m.role as 'user' | 'assistant',
      content: [{ text: m.content }],
    }));

  const command = new ConverseCommand({
    modelId: modelConfig.modelId,
    messages: conversationMessages,
    ...(systemTexts.length > 0 ? { system: systemTexts } : {}),
    inferenceConfig: {
      maxTokens: options?.maxTokens ?? 4096,
      temperature: options?.temperature ?? 0.7,
    },
  });

  const response = await client.send(command);

  const content =
    response.output?.message?.content?.[0]?.text ?? '';
  const inputTokens = response.usage?.inputTokens ?? 0;
  const outputTokens = response.usage?.outputTokens ?? 0;

  return { content, inputTokens, outputTokens };
}

/**
 * Call OpenAI Chat Completions API (non-streaming).
 */
async function callOpenAI(
  modelConfig: ModelConfig,
  messages: LLMMessage[],
  options?: LLMCallOptions,
): Promise<LLMResponse> {
  const client = getOpenAIClient();

  const response = await client.chat.completions.create({
    model: modelConfig.modelId,
    messages: messages.map((m) => ({
      role: m.role,
      content: m.content,
    })),
    max_tokens: options?.maxTokens ?? 4096,
    temperature: options?.temperature ?? 0.7,
  });

  const content = response.choices[0]?.message?.content ?? '';
  const inputTokens = response.usage?.prompt_tokens ?? 0;
  const outputTokens = response.usage?.completion_tokens ?? 0;

  return { content, inputTokens, outputTokens };
}

/**
 * Call Google Gemini GenAI API (non-streaming).
 * System instructions are extracted and passed via `config.systemInstruction`.
 * Remaining messages are concatenated into the `contents` field.
 */
async function callGemini(
  modelConfig: ModelConfig,
  messages: LLMMessage[],
  options?: LLMCallOptions,
): Promise<LLMResponse> {
  const client = getGeminiClient();

  // Extract system instruction (concatenate all system messages)
  const systemParts = messages
    .filter((m) => m.role === 'system')
    .map((m) => m.content);
  const systemInstruction =
    systemParts.length > 0 ? systemParts.join('\n\n') : undefined;

  // Build contents array for non-system messages
  const contents = messages
    .filter((m) => m.role !== 'system')
    .map((m) => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }],
    }));

  const response = await client.models.generateContent({
    model: modelConfig.modelId,
    contents,
    config: {
      ...(systemInstruction ? { systemInstruction } : {}),
      maxOutputTokens: options?.maxTokens ?? 4096,
      temperature: options?.temperature ?? 0.7,
    },
  });

  const content = response.text ?? '';
  const inputTokens = response.usageMetadata?.promptTokenCount ?? 0;
  const outputTokens = response.usageMetadata?.candidatesTokenCount ?? 0;

  return { content, inputTokens, outputTokens };
}

// ── Main entry point ──────────────────────────────────────────────────────

/**
 * Call an LLM by model ID. Resolves the model config from the registry,
 * routes to the correct provider, and returns the response text with
 * token usage.
 *
 * @throws if the model ID is not found or the provider call fails.
 */
export async function callLLM(
  modelId: string,
  messages: LLMMessage[],
  options?: LLMCallOptions,
): Promise<LLMResponse> {
  const modelConfig = AVAILABLE_MODELS.find((m) => m.id === modelId);
  if (!modelConfig) {
    throw new Error(
      `Unknown model ID "${modelId}". Available: ${AVAILABLE_MODELS.map((m) => m.id).join(', ')}`,
    );
  }

  switch (modelConfig.provider) {
    case 'bedrock': {
      try {
        return await callBedrock(modelConfig, messages, options);
      } catch (err) {
        throw new Error(
          `Bedrock provider error (model: ${modelConfig.modelId}): ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }

    case 'openai': {
      try {
        return await callOpenAI(modelConfig, messages, options);
      } catch (err) {
        throw new Error(
          `OpenAI provider error (model: ${modelConfig.modelId}): ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }

    case 'gemini': {
      try {
        return await callGemini(modelConfig, messages, options);
      } catch (err) {
        throw new Error(
          `Gemini provider error (model: ${modelConfig.modelId}): ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }

    default: {
      const _exhaustive: never = modelConfig.provider;
      throw new Error(`Unsupported provider: ${_exhaustive}`);
    }
  }
}

// ── Utility exports ───────────────────────────────────────────────────────

/** Look up a model config by ID, or return undefined. */
export function getModelConfig(modelId: string): ModelConfig | undefined {
  return AVAILABLE_MODELS.find((m) => m.id === modelId);
}

/** Get all models for a given provider. */
export function getModelsByProvider(
  provider: ModelConfig['provider'],
): ModelConfig[] {
  return AVAILABLE_MODELS.filter((m) => m.provider === provider);
}
