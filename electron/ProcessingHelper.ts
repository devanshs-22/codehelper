// ProcessingHelper.ts
import fs from "node:fs"
import path from "node:path"
import { ScreenshotHelper } from "./ScreenshotHelper"
import { IProcessingHelperDeps } from "./main"
import * as axios from "axios"
import { app, BrowserWindow, dialog } from "electron"
import { OpenAI } from "openai"
import { configHelper } from "./ConfigHelper"
import Anthropic from '@anthropic-ai/sdk';

// Interface for Gemini API requests
interface GeminiMessage {
  role: string;
  parts: Array<{
    text?: string;
    inlineData?: {
      mimeType: string;
      data: string;
    }
  }>;
}

interface GeminiResponse {
  candidates: Array<{
    content: {
      parts: Array<{
        text: string;
      }>;
    };
    finishReason: string;
  }>;
}
interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: Array<{
    type: 'text' | 'image';
    text?: string;
    source?: {
      type: 'base64';
      media_type: string;
      data: string;
    };
  }>;
}
export class ProcessingHelper {
  private deps: IProcessingHelperDeps
  private screenshotHelper: ScreenshotHelper
  private openaiClient: OpenAI | null = null
  private geminiApiKey: string | null = null
  private anthropicClient: Anthropic | null = null

  // AbortControllers for API requests
  private currentProcessingAbortController: AbortController | null = null
  private currentExtraProcessingAbortController: AbortController | null = null

  constructor(deps: IProcessingHelperDeps) {
    this.deps = deps
    this.screenshotHelper = deps.getScreenshotHelper()
    
    // Initialize AI client based on config
    this.initializeAIClient();
    
    // Listen for config changes to re-initialize the AI client
    configHelper.on('config-updated', () => {
      this.initializeAIClient();
    });
  }
  
  /**
   * Initialize or reinitialize the AI client with current config
   */
  private initializeAIClient(): void {
    try {
      const config = configHelper.loadConfig();
      
      if (config.apiProvider === "openai") {
        if (config.apiKey) {
          this.openaiClient = new OpenAI({ 
            apiKey: config.apiKey,
            timeout: 120000, // 120 second timeout
            maxRetries: 2
          });
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.log("OpenAI client initialized successfully");
        } else {
          this.openaiClient = null;
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.warn("No API key available, OpenAI client not initialized");
        }
      } else if (config.apiProvider === "gemini"){
        // Gemini client initialization
        this.openaiClient = null;
        this.anthropicClient = null;
        if (config.apiKey) {
          this.geminiApiKey = config.apiKey;
          console.log("Gemini API key set successfully");
        } else {
          this.openaiClient = null;
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.warn("No API key available, Gemini client not initialized");
        }
      } else if (config.apiProvider === "anthropic") {
        // Reset other clients
        this.openaiClient = null;
        this.geminiApiKey = null;
        if (config.apiKey) {
          this.anthropicClient = new Anthropic({
            apiKey: config.apiKey,
            timeout: 120000,
            maxRetries: 2
          });
          console.log("Anthropic client initialized successfully");
        } else {
          this.openaiClient = null;
          this.geminiApiKey = null;
          this.anthropicClient = null;
          console.warn("No API key available, Anthropic client not initialized");
        }
      }
    } catch (error) {
      console.error("Failed to initialize AI client:", error);
      this.openaiClient = null;
      this.geminiApiKey = null;
      this.anthropicClient = null;
    }
  }

  private async waitForInitialization(
    mainWindow: BrowserWindow
  ): Promise<void> {
    let attempts = 0
    const maxAttempts = 50 // 5 seconds total

    while (attempts < maxAttempts) {
      const isInitialized = await mainWindow.webContents.executeJavaScript(
        "window.__IS_INITIALIZED__"
      )
      if (isInitialized) return
      await new Promise((resolve) => setTimeout(resolve, 100))
      attempts++
    }
    throw new Error("App failed to initialize after 5 seconds")
  }

  private async getCredits(): Promise<number> {
    const mainWindow = this.deps.getMainWindow()
    if (!mainWindow) return 999 // Unlimited credits in this version

    try {
      await this.waitForInitialization(mainWindow)
      return 999 // Always return sufficient credits to work
    } catch (error) {
      console.error("Error getting credits:", error)
      return 999 // Unlimited credits as fallback
    }
  }

  private async getLanguage(): Promise<string> {
    try {
      const config = configHelper.loadConfig();
      if (config.language) {
        return config.language;
      }
      
      const mainWindow = this.deps.getMainWindow()
      if (mainWindow) {
        try {
          await this.waitForInitialization(mainWindow)
          const language = await mainWindow.webContents.executeJavaScript(
            "window.__LANGUAGE__"
          )

          if (
            typeof language === "string" &&
            language !== undefined &&
            language !== null
          ) {
            return language;
          }
        } catch (err) {
          console.warn("Could not get language from window", err);
        }
      }
      
      return "python";
    } catch (error) {
      console.error("Error getting language:", error)
      return "python"
    }
  }

  public async processScreenshots(): Promise<void> {
    const mainWindow = this.deps.getMainWindow()
    if (!mainWindow) return

    const config = configHelper.loadConfig();
    
    if (config.apiProvider === "openai" && !this.openaiClient) {
      this.initializeAIClient();
      if (!this.openaiClient) {
        console.error("OpenAI client not initialized");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.API_KEY_INVALID);
        return;
      }
    } else if (config.apiProvider === "gemini" && !this.geminiApiKey) {
      this.initializeAIClient();
      if (!this.geminiApiKey) {
        console.error("Gemini API key not initialized");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.API_KEY_INVALID);
        return;
      }
    } else if (config.apiProvider === "anthropic" && !this.anthropicClient) {
      this.initializeAIClient();
      if (!this.anthropicClient) {
        console.error("Anthropic client not initialized");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.API_KEY_INVALID);
        return;
      }
    }

    const view = this.deps.getView()
    console.log("Processing screenshots in view:", view)

    if (view === "queue") {
      mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.INITIAL_START)
      const screenshotQueue = this.screenshotHelper.getScreenshotQueue()
      
      if (!screenshotQueue || screenshotQueue.length === 0) {
        console.log("No screenshots found in queue");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }

      const existingScreenshots = screenshotQueue.filter(path => fs.existsSync(path));
      if (existingScreenshots.length === 0) {
        console.log("Screenshot files don't exist on disk");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }

      try {
        this.currentProcessingAbortController = new AbortController()
        const { signal } = this.currentProcessingAbortController

        const screenshots = await Promise.all(
          existingScreenshots.map(async (path) => {
            try {
              return {
                path,
                preview: await this.screenshotHelper.getImagePreview(path),
                data: fs.readFileSync(path).toString('base64')
              };
            } catch (err) {
              console.error(`Error reading screenshot ${path}:`, err);
              return null;
            }
          })
        )

        const validScreenshots = screenshots.filter(Boolean);
        
        if (validScreenshots.length === 0) {
          throw new Error("Failed to load screenshot data");
        }

        const result = await this.processScreenshotsHelper(validScreenshots, signal)

        if (!result.success) {
          console.log("Processing failed:", result.error)
          if (result.error?.includes("API Key") || result.error?.includes("OpenAI") || result.error?.includes("Gemini")) {
            mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.API_KEY_INVALID)
          } else {
            mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR, result.error)
          }
          this.deps.setView("queue")
          return
        }

        console.log("Setting view to solutions after successful processing")
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.SOLUTION_SUCCESS, result.data)
        this.deps.setView("solutions")
      } catch (error: any) {
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR, error)
        console.error("Processing error:", error)
        if (axios.isCancel(error)) {
          mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR, "Processing was canceled by the user.")
        } else {
          mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.INITIAL_SOLUTION_ERROR, error.message || "Server error. Please try again.")
        }
        this.deps.setView("queue")
      } finally {
        this.currentProcessingAbortController = null
      }
    } else {
      const extraScreenshotQueue = this.screenshotHelper.getExtraScreenshotQueue()
      
      if (!extraScreenshotQueue || extraScreenshotQueue.length === 0) {
        console.log("No extra screenshots found in queue");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }

      const existingExtraScreenshots = extraScreenshotQueue.filter(path => fs.existsSync(path));
      if (existingExtraScreenshots.length === 0) {
        console.log("Extra screenshot files don't exist on disk");
        mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS);
        return;
      }
      
      mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.DEBUG_START)

      this.currentExtraProcessingAbortController = new AbortController()
      const { signal } = this.currentExtraProcessingAbortController

      try {
        const allPaths = [
          ...this.screenshotHelper.getScreenshotQueue(),
          ...existingExtraScreenshots
        ];
        
        const screenshots = await Promise.all(
          allPaths.map(async (path) => {
            try {
              if (!fs.existsSync(path)) {
                console.warn(`Screenshot file does not exist: ${path}`);
                return null;
              }
              
              return {
                path,
                preview: await this.screenshotHelper.getImagePreview(path),
                data: fs.readFileSync(path).toString('base64')
              };
            } catch (err) {
              console.error(`Error reading screenshot ${path}:`, err);
              return null;
            }
          })
        )
        
        const validScreenshots = screenshots.filter(Boolean);
        
        if (validScreenshots.length === 0) {
          throw new Error("Failed to load screenshot data for debugging");
        }
        
        const result = await this.processExtraScreenshotsHelper(validScreenshots, signal)

        if (result.success) {
          this.deps.setHasDebugged(true)
          mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.DEBUG_SUCCESS, result.data)
        } else {
          mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.DEBUG_ERROR, result.error)
        }
      } catch (error: any) {
        if (axios.isCancel(error)) {
          mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.DEBUG_ERROR, "Extra processing was canceled by the user.")
        } else {
          mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.DEBUG_ERROR, error.message)
        }
      } finally {
        this.currentExtraProcessingAbortController = null
      }
    }
  }

  private async processScreenshotsHelper(
    screenshots: Array<{ path: string; data: string }>,
    signal: AbortSignal
  ) {
    try {
      const config = configHelper.loadConfig();
      const language = await this.getLanguage();
      const mainWindow = this.deps.getMainWindow();
      
      const imageDataList = screenshots.map(screenshot => screenshot.data);
      
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Classifying problem type...",
          progress: 10
        });
      }
      
      const classificationPrompt = "Is this image primarily a coding problem or a multiple-choice question (MCQ)? Respond with only one word: 'coding' or 'mcq'.";

      const classificationMessages: GeminiMessage[] = [
        {
          role: "user",
          parts: [
            { text: classificationPrompt },
            ...imageDataList.map(data => ({
              inlineData: {
                mimeType: "image/png",
                data: data
              }
            }))
          ]
        }
      ];

      let requestType = 'coding'; 

      try {
        const classificationResponse = await axios.default.post(
          `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=${this.geminiApiKey}`,
          {
            contents: classificationMessages,
            generationConfig: {
              temperature: 0.0,
              maxOutputTokens: 10
            }
          },
          { signal, timeout: 30000 }
        );
        
        const classificationText = classificationResponse.data.candidates[0].content.parts[0].text.toLowerCase().trim();
        if (classificationText.includes('mcq')) {
          requestType = 'mcq';
        }
        console.log(`Request classified as: ${requestType}`);

      } catch (error) {
        console.error("Error during classification, defaulting to 'coding'.", error);
      }

      if (requestType === 'mcq') {
        if (mainWindow) {
          mainWindow.webContents.send("processing-status", {
            message: "Analyzing MCQ...",
            progress: 50
          });
        }

        const mcqPrompt = "Analyze the multiple-choice question in the image and provide the correct answer. First, state the correct option (e.g., 'C'). Then, provide a concise but clear explanation for why that answer is correct and the others are incorrect. Format the response cleanly.";

        const mcqMessages: GeminiMessage[] = [
          {
            role: "user",
            parts: [
              { text: mcqPrompt },
              ...imageDataList.map(data => ({
                inlineData: {
                  mimeType: "image/png",
                  data: data
                }
              }))
            ]
          }
        ];

        const response = await axios.default.post(
          `https://generativelanguage.googleapis.com/v1beta/models/${config.solutionModel || "gemini-2.5-pro"}:generateContent?key=${this.geminiApiKey}`,
          {
            contents: mcqMessages,
            generationConfig: {
              temperature: 0.2,
              maxOutputTokens: 8000
            }
          },
          { signal, timeout: 120000 }
        );
        const responseText = response.data.candidates[0].content.parts[0].text;
        
        const mcqSolution = {
          code: `Correct Answer & Explanation:\n\n${responseText}`,
          thoughts: ["Analyzed the MCQ to determine the correct option.", "Provided a step-by-step explanation."],
          time_complexity: "N/A",
          space_complexity: "N/A"
        };
        
        return { success: true, data: mcqSolution };

      } else {
        // Handle 'coding' request type
        let problemInfo;
        
        if (config.apiProvider === "openai") {
          // Verify OpenAI client
          if (!this.openaiClient) {
            this.initializeAIClient();
            if (!this.openaiClient) {
              return { success: false, error: "OpenAI API key not configured or invalid. Please check your settings." };
            }
          }

          const messages = [
            {
              role: "system" as const, 
              content: "You are a coding challenge interpreter. Analyze the screenshot of the coding problem and extract all relevant information. Return the information in JSON format with these fields: problem_statement, constraints, example_input, example_output. Just return the structured JSON without any other text."
            },
            {
              role: "user" as const,
              content: [
                {
                  type: "text" as const, 
                  text: `Extract the coding problem details from these screenshots. Return in JSON format. Preferred coding language we gonna use for this problem is ${language}.`
                },
                ...imageDataList.map(data => ({
                  type: "image_url" as const,
                  image_url: { url: `data:image/png;base64,${data}` }
                }))
              ]
            }
          ];

          const extractionResponse = await this.openaiClient.chat.completions.create({
            model: config.extractionModel || "gpt-4o",
            messages: messages,
            max_tokens: 4000,
            temperature: 0.2
          });

          try {
            const responseText = extractionResponse.choices[0].message.content;
            const jsonText = responseText.replace(/```json|```/g, '').trim();
            problemInfo = JSON.parse(jsonText);
          } catch (error) {
            console.error("Error parsing OpenAI response:", error);
            return { success: false, error: "Failed to parse problem information. Please try again or use clearer screenshots." };
          }
        } else if (config.apiProvider === "gemini")  {
          if (!this.geminiApiKey) {
            return { success: false, error: "Gemini API key not configured. Please check your settings." };
          }

          try {
            const geminiMessages: GeminiMessage[] = [
              {
                role: "user",
                parts: [
                  {
                    text: `You are a coding challenge interpreter. Analyze the screenshots of the coding problem and extract all relevant information. Return the information in JSON format with these fields: problem_statement, constraints, example_input, example_output. Just return the structured JSON without any other text. Preferred coding language we gonna use for this problem is ${language}.`
                  },
                  ...imageDataList.map(data => ({
                    inlineData: {
                      mimeType: "image/png",
                      data: data
                    }
                  }))
                ]
              }
            ];

            const response = await axios.default.post(
              `https://generativelanguage.googleapis.com/v1beta/models/${config.extractionModel || "gemini-2.0-flash"}:generateContent?key=${this.geminiApiKey}`,
              {
                contents: geminiMessages,
                generationConfig: {
                  temperature: 0.2,
                  maxOutputTokens: 20000
                }
              },
              { signal, timeout: 120000 }
            );

            const responseData = response.data as GeminiResponse;
            
            if (!responseData.candidates || responseData.candidates.length === 0) {
              throw new Error("Empty response from Gemini API");
            }
            
            const responseText = responseData.candidates[0].content.parts[0].text;
            const jsonText = responseText.replace(/```json|```/g, '').trim();
            problemInfo = JSON.parse(jsonText);
          } catch (error) {
            console.error("Error using Gemini API:", error);
            return { success: false, error: "Failed to process with Gemini API. Please check your API key or try again later." };
          }
        } else if (config.apiProvider === "anthropic") {
          if (!this.anthropicClient) {
            return { success: false, error: "Anthropic API key not configured. Please check your settings." };
          }

          try {
            const messages = [
              {
                role: "user" as const,
                content: [
                  {
                    type: "text" as const,
                    text: `Extract the coding problem details from these screenshots. Return in JSON format with these fields: problem_statement, constraints, example_input, example_output. Preferred coding language is ${language}.`
                  },
                  ...imageDataList.map(data => ({
                    type: "image" as const,
                    source: {
                      type: "base64" as const,
                      media_type: "image/png" as const,
                      data: data
                    }
                  }))
                ]
              }
            ];

            const response = await this.anthropicClient.messages.create({
              model: config.extractionModel || "claude-3-7-sonnet-20250219",
              max_tokens: 4000,
              messages: messages,
              temperature: 0.2
            });

            const responseText = (response.content[0] as { type: 'text', text: string }).text;
            const jsonText = responseText.replace(/```json|```/g, '').trim();
            problemInfo = JSON.parse(jsonText);
          } catch (error: any) {
            console.error("Error using Anthropic API:", error);
            if (error.status === 429) {
              return { success: false, error: "Claude API rate limit exceeded. Please wait a few minutes before trying again." };
            } else if (error.status === 413 || (error.message && error.message.includes("token"))) {
              return { success: false, error: "Your screenshots contain too much information for Claude to process. Switch to OpenAI or Gemini in settings which can handle larger inputs." };
            }
            return { success: false, error: "Failed to process with Anthropic API. Please check your API key or try again later." };
          }
        }
        
        if (mainWindow) {
          mainWindow.webContents.send("processing-status", {
            message: "Problem analyzed successfully. Preparing to generate solution...",
            progress: 40
          });
        }

        this.deps.setProblemInfo(problemInfo);

        if (mainWindow) {
          mainWindow.webContents.send(
            this.deps.PROCESSING_EVENTS.PROBLEM_EXTRACTED,
            problemInfo
          );

          const solutionsResult = await this.generateSolutionsHelper(signal);
          if (solutionsResult.success) {
            this.screenshotHelper.clearExtraScreenshotQueue();
            return { success: true, data: solutionsResult.data };
          } else {
            throw new Error(
              solutionsResult.error || "Failed to generate solutions"
            );
          }
        }
        return { success: false, error: "Failed to process screenshots" };
      }
    } catch (error: any) {
      if (axios.isCancel(error)) {
        return { success: false, error: "Processing was canceled by the user." };
      }
      
      if (error?.response?.status === 401) {
        return { success: false, error: "Invalid OpenAI API key. Please check your settings." };
      } else if (error?.response?.status === 429) {
        return { success: false, error: "OpenAI API rate limit exceeded or insufficient credits. Please try again later." };
      } else if (error?.response?.status === 500) {
        return { success: false, error: "OpenAI server error. Please try again later." };
      }

      console.error("API Error Details:", error);
      return { 
        success: false, 
        error: error.message || "Failed to process screenshots. Please try again." 
      };
    }
  }

  private async generateSolutionsHelper(signal: AbortSignal) {
    try {
      const problemInfo = this.deps.getProblemInfo();
      const language = await this.getLanguage();
      const config = configHelper.loadConfig();
      const mainWindow = this.deps.getMainWindow();

      if (!problemInfo) {
        throw new Error("No problem info available");
      }

      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Creating optimal solution with detailed explanations...",
          progress: 60
        });
      }

      // In generateSolutionsHelper function
      const promptText = `
      You are a skilled software engineer tasked with solving a problem from an online assessment. 
      Your goal is to provide a clean, efficient, and well-structured solution which should work on given constraints.

      PROBLEM CONTEXT:
      - Statement: ${problemInfo.problem_statement}
      - Constraints: ${problemInfo.constraints || "No specific constraints provided."}
      - Example Input: ${problemInfo.example_input || "No example input provided."}
      - Example Output: ${problemInfo.example_output || "No example output provided."}
      - Language: ${language}

      INSTRUCTIONS:
      1.  **Generate an Optimal Solution:** Provide an efficient solution that is suitable for the given constraints. Prioritize standard, well-known optimal algorithms (e.g., using hashmaps, two-pointers, sieve of erasthnis, monotonic stack ,sorting, dynamic programming) over complex approaches, if constraints are very high then only use complex and clever approaches.
      2.  **Code is the Priority:** Your main output should be the complete and correct source code.
      3.  **Use Inline Comments:** Explain any complex parts of your logic with brief comments inside the code. Avoid long, separate explanation paragraphs.
      4.  **Provide Complexity:** After the code block, state the Time and Space Complexity on separate lines.
      `;

      let responseContent;
      
      if (config.apiProvider === "openai") {
        if (!this.openaiClient) {
          return {
            success: false,
            error: "OpenAI API key not configured. Please check your settings."
          };
        }
        
        const solutionResponse = await this.openaiClient.chat.completions.create({
          model: config.solutionModel || "gpt-4o",
          messages: [
            { role: "system", content: "You are an expert coding interview assistant. Provide clear, optimal solutions with detailed explanations." },
            { role: "user", content: promptText }
          ],
          max_tokens: 50000,
          temperature: 0.2
        });

        responseContent = solutionResponse.choices[0].message.content;
      } else if (config.apiProvider === "gemini")  {
        if (!this.geminiApiKey) {
          return {
            success: false,
            error: "Gemini API key not configured. Please check your settings."
          };
        }
        
        try {
          const geminiMessages = [
            {
              role: "user",
              parts: [
                {
                  text: `You are an expert coding interview assistant. Provide a clear, optimal solution  for this problem:\n\n${promptText}`
                }
              ]
            }
          ];
          
          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/${config.solutionModel || "gemini-2.0-flash"}:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: {
                temperature: 0.2,
                maxOutputTokens: 50000 
              }
            },
            { signal, timeout: 180000 }
          );

          const responseData = response.data as GeminiResponse;
          
          if (!responseData.candidates || responseData.candidates.length === 0) {
            throw new Error("Empty response from Gemini API");
          }
          
          responseContent = responseData.candidates[0].content.parts[0].text;
        } catch (error) {
          console.error("Error using Gemini API for solution:", error);
          return {
            success: false,
            error: "Failed to generate solution with Gemini API. Please check your API key or try again later."
          };
        }
      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) {
          return {
            success: false,
            error: "Anthropic API key not configured. Please check your settings."
          };
        }
        
        try {
          const messages = [
            {
              role: "user" as const,
              content: [
                {
                  type: "text" as const,
                  text: `You are an expert coding interview assistant. Provide a clear, optimal solution with detailed explanations for this problem:\n\n${promptText}`
                }
              ]
            }
          ];

          const response = await this.anthropicClient.messages.create({
            model: config.solutionModel || "claude-3-7-sonnet-20250219",
            max_tokens: 50000,
            messages: messages,
            temperature: 0.2
          });

          responseContent = (response.content[0] as { type: 'text', text: string }).text;
        } catch (error: any) {
          console.error("Error using Anthropic API for solution:", error);
          if (error.status === 429) {
            return { success: false, error: "Claude API rate limit exceeded. Please wait a few minutes before trying again." };
          } else if (error.status === 413 || (error.message && error.message.includes("token"))) {
            return { success: false, error: "Your screenshots contain too much information for Claude to process. Switch to OpenAI or Gemini in settings which can handle larger inputs." };
          }
          return { success: false, error: "Failed to generate solution with Anthropic API. Please check your API key or try again later." };
        }
      }
      
      const codeMatch = responseContent.match(/```(?:\w+)?\s*([\s\S]*?)```/);
      const code = codeMatch ? codeMatch[1].trim() : responseContent;
      
      const thoughtsRegex = /(?:Thoughts:|Key Insights:|Reasoning:|Approach:)([\s\S]*?)(?:Time complexity:|$)/i;
      const thoughtsMatch = responseContent.match(thoughtsRegex);
      let thoughts: string[] = [];
      
      if (thoughtsMatch && thoughtsMatch[1]) {
        const bulletPoints = thoughtsMatch[1].match(/(?:^|\n)\s*(?:[-*•]|\d+\.)\s*(.*)/g);
        if (bulletPoints) {
          thoughts = bulletPoints.map(point => 
            point.replace(/^\s*(?:[-*•]|\d+\.)\s*/, '').trim()
          ).filter(Boolean);
        } else {
          thoughts = thoughtsMatch[1].split('\n')
            .map((line) => line.trim())
            .filter(Boolean);
        }
      }
      
      const timeComplexityPattern = /Time complexity:?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\s*(?:Space complexity|$))/i;
      const spaceComplexityPattern = /Space complexity:?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\s*(?:[A-Z]|$))/i;
      
      let timeComplexity = "O(n) - See explanation in 'Your Thoughts'.";
      let spaceComplexity = "O(n) - See explanation in 'Your Thoughts'.";
      
      const timeMatch = responseContent.match(timeComplexityPattern);
      if (timeMatch && timeMatch[1]) {
        timeComplexity = timeMatch[1].trim();
      }
      
      const spaceMatch = responseContent.match(spaceComplexityPattern);
      if (spaceMatch && spaceMatch[1]) {
        spaceComplexity = spaceMatch[1].trim();
      }

      const formattedResponse = {
        code: code,
        thoughts: thoughts.length > 0 ? thoughts : ["Solution approach based on efficiency and readability"],
        time_complexity: timeComplexity,
        space_complexity: spaceComplexity
      };

      return { success: true, data: formattedResponse };
    } catch (error: any) {
      if (axios.isCancel(error)) {
        return { success: false, error: "Processing was canceled by the user." };
      }
      
      if (error?.response?.status === 401) {
        return { success: false, error: "Invalid OpenAI API key. Please check your settings." };
      } else if (error?.response?.status === 429) {
        return { success: false, error: "OpenAI API rate limit exceeded or insufficient credits. Please try again later." };
      }
      
      console.error("Solution generation error:", error);
      return { success: false, error: error.message || "Failed to generate solution" };
    }
  }

  private async processExtraScreenshotsHelper(
    screenshots: Array<{ path: string; data: string }>,
    signal: AbortSignal
  ) {
    try {
      const problemInfo = this.deps.getProblemInfo();
      const language = await this.getLanguage();
      const config = configHelper.loadConfig();
      const mainWindow = this.deps.getMainWindow();

      if (!problemInfo) {
        throw new Error("No problem info available");
      }

      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Processing debug screenshots...",
          progress: 30
        });
      }

      const imageDataList = screenshots.map(screenshot => screenshot.data);
      
      let debugContent;
      
      if (config.apiProvider === "openai") {
        if (!this.openaiClient) {
          return { success: false, error: "OpenAI API key not configured. Please check your settings." };
        }
        
        const messages = [
          {
            role: "system" as const, 
            content: `You are a coding interview assistant helping debug and improve solutions. Analyze these screenshots which include either error messages, incorrect outputs, or test cases, and provide detailed debugging help.

Your response MUST follow this exact structure with these section headers (use ### for headers):
### Issues Identified
- List each issue as a bullet point with clear explanation

### Specific Improvements and Corrections
- List specific code changes needed as bullet points

### Optimizations
- List any performance optimizations if applicable

### Explanation of Changes Needed
Here provide a clear explanation of why the changes are needed

### Key Points
- Summary bullet points of the most important takeaways

If you include code examples, use proper markdown code blocks with language specification (e.g. \`\`\`java).`
          },
          {
            role: "user" as const,
            content: [
              {
                type: "text" as const, 
                text: `I'm solving this coding problem: "${problemInfo.problem_statement}" in ${language}. I need help with debugging or improving my solution. Here are screenshots of my code, the errors or test cases. Please provide a detailed analysis with:
1. What issues you found in my code
2. Specific improvements and corrections
3. Any optimizations that would make the solution better
4. A clear explanation of the changes needed` 
              },
              ...imageDataList.map(data => ({
                type: "image_url" as const,
                image_url: { url: `data:image/png;base64,${data}` }
              }))
            ]
          }
        ];

        if (mainWindow) {
          mainWindow.webContents.send("processing-status", {
            message: "Analyzing code and generating debug feedback...",
            progress: 60
          });
        }

        const debugResponse = await this.openaiClient.chat.completions.create({
          model: config.debuggingModel || "gpt-4o",
          messages: messages,
          max_tokens: 4000,
          temperature: 0.2
        });
        
        debugContent = debugResponse.choices[0].message.content;
      } else if (config.apiProvider === "gemini")  {
        if (!this.geminiApiKey) {
          return { success: false, error: "Gemini API key not configured. Please check your settings." };
        }
        
        try {
          // In processExtraScreenshotsHelper function
      const debugPrompt = `
      You are an expert code debugger. A user has provided screenshots of their code that has bugs, errors, or fails test cases.
      Your primary goal is to provide the fully corrected, working code.

      PROBLEM CONTEXT:
      - The original problem statement is: "${problemInfo.problem_statement}"
      - The programming language is: ${language}

      INSTRUCTIONS:
      1.  **Provide the Corrected Code First:** Immediately provide the complete, corrected, and working version of the user's code inside a single markdown code block.
      2.  **Explain the Fixes:** After the code block, create a "Changes Made" section. In this section, use a brief bulleted list to explain exactly what you fixed and why.
      `;

          const geminiMessages = [
            {
              role: "user",
              parts: [
                { text: debugPrompt },
                ...imageDataList.map(data => ({
                  inlineData: {
                    mimeType: "image/png",
                    data: data
                  }
                }))
              ]
            }
          ];

          if (mainWindow) {
            mainWindow.webContents.send("processing-status", {
              message: "Analyzing code and generating debug feedback with Gemini...",
              progress: 60
            });
          }

          const response = await axios.default.post(
            `https://generativelanguage.googleapis.com/v1beta/models/${config.debuggingModel || "gemini-2.0-flash"}:generateContent?key=${this.geminiApiKey}`,
            {
              contents: geminiMessages,
              generationConfig: {
                temperature: 0.2,
                maxOutputTokens: 50000
              }
            },
            { signal,timeout: 180000 }
          );

          const responseData = response.data as GeminiResponse;
          
          if (!responseData.candidates || responseData.candidates.length === 0) {
            throw new Error("Empty response from Gemini API");
          }
          
          debugContent = responseData.candidates[0].content.parts[0].text;
        } catch (error) {
          console.error("Error using Gemini API for debugging:", error);
          return { success: false, error: "Failed to process debug request with Gemini API. Please check your API key or try again later." };
        }
      } else if (config.apiProvider === "anthropic") {
        if (!this.anthropicClient) {
          return { success: false, error: "Anthropic API key not configured. Please check your settings." };
        }
        
        try {
          const debugPrompt = `
You are a coding interview assistant helping debug and improve solutions. Analyze these screenshots which include either error messages, incorrect outputs, or test cases, and provide detailed debugging help.

I'm solving this coding problem: "${problemInfo.problem_statement}" in ${language}. I need help with debugging or improving my solution.

YOUR RESPONSE MUST FOLLOW THIS EXACT STRUCTURE WITH THESE SECTION HEADERS:
### Issues Identified
- List each issue as a bullet point with clear explanation

### Specific Improvements and Corrections
- List specific code changes needed as bullet points

### Optimizations
- List any performance optimizations if applicable

### Explanation of Changes Needed
Here provide a clear explanation of why the changes are needed

### Key Points
- Summary bullet points of the most important takeaways

If you include code examples, use proper markdown code blocks with language specification.
`;

          const messages = [
            {
              role: "user" as const,
              content: [
                {
                  type: "text" as const,
                  text: debugPrompt
                },
                ...imageDataList.map(data => ({
                  type: "image" as const,
                  source: {
                    type: "base64" as const,
                    media_type: "image/png" as const, 
                    data: data
                  }
                }))
              ]
            }
          ];

          if (mainWindow) {
            mainWindow.webContents.send("processing-status", {
              message: "Analyzing code and generating debug feedback with Claude...",
              progress: 60
            });
          }

          const response = await this.anthropicClient.messages.create({
            model: config.debuggingModel || "claude-3-7-sonnet-20250219",
            max_tokens: 4000,
            messages: messages,
            temperature: 0.2
          });
          
          debugContent = (response.content[0] as { type: 'text', text: string }).text;
        } catch (error: any) {
          console.error("Error using Anthropic API for debugging:", error);
          if (error.status === 429) {
            return { success: false, error: "Claude API rate limit exceeded. Please wait a few minutes before trying again." };
          } else if (error.status === 413 || (error.message && error.message.includes("token"))) {
            return { success: false, error: "Your screenshots contain too much information for Claude to process. Switch to OpenAI or Gemini in settings which can handle larger inputs." };
          }
          return { success: false, error: "Failed to process debug request with Anthropic API. Please check your API key or try again later." };
        }
      }
      
      if (mainWindow) {
        mainWindow.webContents.send("processing-status", {
          message: "Debug analysis complete",
          progress: 100
        });
      }

      let extractedCode = "// Debug mode - see analysis below";
      const codeMatch = debugContent.match(/```(?:[a-zA-Z]+)?([\s\S]*?)```/);
      if (codeMatch && codeMatch[1]) {
        extractedCode = codeMatch[1].trim();
      }

      let formattedDebugContent = debugContent;
      
      if (!debugContent.includes('# ') && !debugContent.includes('## ')) {
        formattedDebugContent = debugContent
          .replace(/issues identified|problems found|bugs found/i, '## Issues Identified')
          .replace(/code improvements|improvements|suggested changes/i, '## Code Improvements')
          .replace(/optimizations|performance improvements/i, '## Optimizations')
          .replace(/explanation|detailed analysis/i, '## Explanation');
      }

      const bulletPoints = formattedDebugContent.match(/(?:^|\n)[ ]*(?:[-*•]|\d+\.)[ ]+([^\n]+)/g);
      const thoughts = bulletPoints 
        ? bulletPoints.map(point => point.replace(/^[ ]*(?:[-*•]|\d+\.)[ ]+/, '').trim()).slice(0, 5)
        : ["Debug analysis based on your screenshots"];
      
      const response = {
        code: extractedCode,
        debug_analysis: formattedDebugContent,
        thoughts: thoughts,
        time_complexity: "N/A - Debug mode",
        space_complexity: "N/A - Debug mode"
      };

      return { success: true, data: response };
    } catch (error: any) {
      console.error("Debug processing error:", error);
      return { success: false, error: error.message || "Failed to process debug request" };
    }
  }

  public cancelOngoingRequests(): void {
    let wasCancelled = false

    if (this.currentProcessingAbortController) {
      this.currentProcessingAbortController.abort()
      this.currentProcessingAbortController = null
      wasCancelled = true
    }

    if (this.currentExtraProcessingAbortController) {
      this.currentExtraProcessingAbortController.abort()
      this.currentExtraProcessingAbortController = null
      wasCancelled = true
    }

    this.deps.setHasDebugged(false)

    this.deps.setProblemInfo(null)

    const mainWindow = this.deps.getMainWindow()
    if (wasCancelled && mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send(this.deps.PROCESSING_EVENTS.NO_SCREENSHOTS)
    }
  }
}