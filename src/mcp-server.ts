import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { OpenAI } from "openai";
import * as z from "zod";
import { promises as fs } from "fs";
import path from "path";
import os from "os";  // 一時ディレクトリのパスを取得するために使用
// import playSound from "play-sound"; // 削除: 再生機能は使わない
import { MCPServerConfig, OpenAIVoice, OpenAIOutputFormat, OpenAITTSModel } from "./types";
import { validateVoice, validateModel, validateFormat, ensureOutputDirectory } from "./utils"; // 追加: ensureOutputDirectory

// // プレイヤーの初期化 // 削除
// const player = playSound({}); // 削除

// ログファイルパス
let logFile = path.join(process.cwd(), 'tts-mcp.log');

/**
 * ログファイルにメッセージを追加します
 * @param {string} message - ログメッセージ
 */
async function logToFile(message: string): Promise<void> {
  const timestamp = new Date().toISOString();
  const logMessage = `${timestamp} - ${message}\n`;
  
  try {
    await fs.appendFile(logFile, logMessage);
  } catch (error) {
    // ログ書き込みエラーは無視（エラー処理のループを避けるため）
  }
}

/**
 * OpenAI APIのクライアントを初期化します
 * @param {string} apiKey - OpenAI APIキー
 * @param {string} [baseUrl] - オプション: APIベースURL
 * @returns {OpenAI} OpenAIクライアントインスタンス
 */
export function initializeClient(apiKey: string, baseUrl?: string): OpenAI {
  if (!apiKey) {
    throw new Error('OpenAI APIキーが設定されていません。--api-keyオプションか環境変数OPENAI_API_KEYを設定してください。');
  }
  
  return new OpenAI({
    apiKey: apiKey,
    ...(baseUrl && { baseURL: baseUrl }) // 追加: baseUrlが存在する場合に設定
  });
}

// オプション名はそのまま利用可能
interface TTSSaveOptions { // 名前変更: Play -> Save (任意だが分かりやすい)
  text: string;
  model: OpenAITTSModel;
  voice: OpenAIVoice;
  speed: number;
  format: OpenAIOutputFormat;
  instructions?: string;
  apiKey: string;
  baseUrl?: string; // 追加: APIベースURL
}

// 結果インターフェースを変更
interface TTSSaveResult {
  filePath: string; // 保存されたファイルパス
  textLength: number;
}

/**
 * テキストを音声に変換してファイルに保存します
 * @param {TTSSaveOptions} options - 変換オプション
 * @returns {Promise<TTSSaveResult>} 処理結果（ファイルパスを含む）
 */
async function textToSpeechAndSave(options: TTSSaveOptions): Promise<TTSSaveResult> { // 名前変更: Play -> Save, 型変更
  const client = initializeClient(options.apiKey, options.baseUrl); // 変更: baseUrlを渡す
  // let tempFilePath: string | null = null; // 削除: 一時ファイルは使わない
  
  try {
    await logToFile('音声生成開始...');
    
    // 入力の検証
    const safeVoice = validateVoice(options.voice);
    const safeModel = validateModel(options.model);
    const safeFormat = validateFormat(options.format);
      
    const response = await client.audio.speech.create({
      model: safeModel,
      voice: safeVoice,
      input: options.text,
      speed: options.speed,
      response_format: safeFormat,
      ...(options.instructions ? { instructions: options.instructions } : {})
    });

    // 音声データを取得
    const buffer = Buffer.from(await response.arrayBuffer());
    
    // 出力ディレクトリとファイルパスを定義
    const outputDir = path.join(process.cwd(), 'output', 'openai-tts-mcp'); // パスを変更
    const outputFilename = `speech_${Date.now()}.${safeFormat}`;
    const outputPath = path.join(outputDir, outputFilename);

    // 出力ディレクトリが存在することを確認
    await ensureOutputDirectory(outputPath); // utilsからインポートした関数を使用

    // バッファをファイルに書き込む
    await fs.writeFile(outputPath, buffer);

    await logToFile(`音声ファイルを作成しました: ${outputPath}`);
    
    // --- 再生ロジック削除 ---
    
    // 結果としてファイルパスとテキスト長を返す
    return {
      filePath: outputPath,
      textLength: options.text.length
    };
  } catch (error: any) {
    // --- 一時ファイル削除ロジック削除 ---

    // APIエラーをより詳細にログ記録
    if (error.response) {
      const apiError = error.response.data;
      await logToFile(`OpenAI API エラー:`);
      await logToFile(`- ステータス: ${error.response.status}`);
      await logToFile(`- メッセージ: ${apiError.error?.message || 'エラー詳細なし'}`);
      await logToFile(`- タイプ: ${apiError.error?.type || 'エラータイプなし'}`);
      
      // 元のエラーもログに記録
      await logToFile(`生のレスポンスデータ: ${JSON.stringify(error.response.data)}`);
    } else {
      // 一般的なエラー
      await logToFile(`エラー: ${error.message}`);
    }
    throw error;
  }
}

/**
 * MCPサーバーを作成して設定します
 * @param {MCPServerConfig} config - サーバー設定
 * @returns {Promise<McpServer>} 設定済みMCPサーバー
 */
async function createMcpServer(config: MCPServerConfig): Promise<McpServer> {
  // カスタムログファイルがあれば設定
  if (config.logFile) {
    logFile = config.logFile;
  }

  // ログファイルを初期化
  await logToFile('---------------------------------------');
  await logToFile(`MCPサーバーを初期化しています...`);
  await logToFile(`設定: モデル=${config.model}, 音声=${config.voice}, フォーマット=${config.format}`);

  // MCPサーバーを作成
  const server = new McpServer({
    name: "tts-mcp",
    version: "1.1.0",
    // サポートする機能を明示的に定義
    capabilities: {
      tools: {}
    }
  });

  // テキスト音声変換と再生ツールのスキーマ定義（Raw Shape）
  const textToSpeechShape = {
    text: z.string().describe("The text content to be converted to speech"),
    speed: z.number().min(0.25).max(4.0).optional().default(1.0).describe("Speech speed factor (0.25 to 4.0, default: 1.0)"),
    instructions: z.string().optional().describe("Optional instructions to guide the speech generation (e.g. emotions, style)"),
  };
  
  // Raw Shapeから型を推論
  type TextToSpeechParams = z.infer<z.ZodObject<typeof textToSpeechShape>>;

  // テキスト音声変換と再生ツールを追加
  server.tool(
    "text-to-speech",
    "Converts text to speech using OpenAI's TTS API and saves it to a file", // 説明を更新
    textToSpeechShape, // Raw Shapeを使用
    async (params: TextToSpeechParams) => { // 型を指定
      try {
        const result = await textToSpeechAndSave({ // 関数名変更
          text: params.text,
          model: config.model,
          voice: config.voice,
          speed: params.speed,
          format: config.format,
          instructions: params.instructions,
          apiKey: config.apiKey,
          baseUrl: config.baseUrl
        });
        
        return {
          content: [ // 応答メッセージを変更
            {
              type: "text",
              text: `音声ファイルを保存しました: ${result.filePath}`
            }
          ],
          metadata: { // メタデータを変更
            file_path: result.filePath,
            text_length: result.textLength
          }
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `エラー: 音声の生成または保存に失敗しました` // エラーメッセージ変更
            },
            {
              type: "text",
              text: `詳細: ${(error as Error).message}`
            },
            {
              type: "text",
              text: `対処方法: OpenAI APIキーの確認、テキスト内容の確認、または別のボイスやモデルの使用をお試しください。`
            }
          ],
          isError: true
        };
      }
    }
  );

  return server;
}

/**
 * MCPサーバーを起動します
 * @param {MCPServerConfig} config - サーバー設定
 * @returns {Promise<void>}
 */
export async function startMcpServer(config: MCPServerConfig): Promise<void> {
  // 入力値の検証
  const safeModel = validateModel(config.model);
  const safeVoice = validateVoice(config.voice);
  const safeFormat = validateFormat(config.format);
  
  // 検証済みの値で設定を上書き
  const validatedConfig: MCPServerConfig = {
    ...config,
    model: safeModel,
    voice: safeVoice,
    format: safeFormat
  };

  // stderr経由でコンテキスト情報を出力
  process.stderr.write(`モデル=${safeModel}, 音声=${safeVoice}, フォーマット=${safeFormat}\n`);

  try {
    // サーバーを作成
    const server = await createMcpServer(validatedConfig);
    
    // STDIOトランスポートを使用
    const transport = new StdioServerTransport();
    
    // サーバーを接続して開始
    await server.connect(transport);
    
    await logToFile("MCPサーバーが起動しました");
  } catch (error) {
    await logToFile(`MCPサーバー起動エラー: ${(error as Error).message}`);
    // stderrにもエラーを出力
    process.stderr.write(`エラー: ${(error as Error).message}\n`);
    throw error;
  }
}
