
import { GoogleGenAI, Type } from "@google/genai";
import { Decision, DecisionResult, Task, ScheduleResult } from "./types";

// Moved GoogleGenAI initialization inside functions as per guidelines to ensure the latest API key is used and that process.env.API_KEY is accessed directly.

export const getDecisionRecommendation = async (decision: Decision): Promise<DecisionResult> => {
  // Initialize AI client right before use
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const model = 'gemini-3-flash-preview';
  
  const dilemmaLength = decision.dilemma.length;
  const hasDetails = decision.pros.length > 0 || decision.cons.length > 0;
  
  const prosText = decision.pros.length > 0 
    ? "User-provided Advantages:\n" + decision.pros.map(p => `- ${p.text}`).join('\n')
    : "No specific advantages provided by user.";
  const consText = decision.cons.length > 0 
    ? "User-provided Disadvantages:\n" + decision.cons.map(c => `- ${c.text}`).join('\n')
    : "No specific disadvantages provided by user.";

  const prompt = `
You are a decision-making AI assistant.

Your task:
1. Read the user’s dilemma: "${decision.dilemma}"
2. Read the list of advantages they provided:
${prosText}
3. Read the list of disadvantages they provided:
${consText}
4. Weigh these factors against each other to determine the best course of action.
5. Provide a clear, decisive recommendation based on this analysis.

${hasDetails || dilemmaLength > 200
  ? "Since this is a detailed dilemma, provide a structured response including a list of core advantages and disadvantages you've identified in your analysis." 
  : "Since this is a short dilemma, be extremely concise and punchy."}

Detect the language of the dilemma and respond in that same language.
`;

  const responseSchema = {
    type: Type.OBJECT,
    properties: {
      recommendation: { type: Type.STRING, description: "A clear, decisive action-oriented recommendation." },
      explanation: { type: Type.STRING, description: "A human-like explanation of the reasoning, showing how you weighed the pros and cons." },
      confidence: { type: Type.NUMBER, description: "Confidence level 0-100." },
      advantages: { 
        type: Type.ARRAY, 
        items: { type: Type.STRING }, 
        description: "Key advantages supporting the recommendation." 
      },
      disadvantages: { 
        type: Type.ARRAY, 
        items: { type: Type.STRING }, 
        description: "Key risks or disadvantages to consider." 
      },
    },
    required: ["recommendation", "explanation", "confidence"]
  };

  try {
    const response = await ai.models.generateContent({
      model,
      contents: prompt,
      config: {
        systemInstruction: `You are 'Decision Helper'. 
        Your goal is to help users make tough choices by analyzing their specific situation.
        1. Read their dilemma.
        2. Analyze their provided pros and cons.
        3. Weigh them logically.
        4. Give a decisive recommendation.
        
        Be friendly and empathetic, but prioritize being DECISIVE. 
        Always respond in the same language as the user's input. 
        In Hebrew, use masculine forms (בלשון זכר).`,
        responseMimeType: "application/json",
        responseSchema,
        temperature: 0.7,
      },
    });

    // Use property text (not a method) and handle potential undefined
    const text = (response.text || "").trim();
    return JSON.parse(text);
  } catch (error) {
    console.error("Gemini API Error:", error);
    throw new Error("My thought process was interrupted. Could you try asking again?");
  }
};

export const breakDownTask = async (taskTitle: string): Promise<string[]> => {
  // Initialize AI client right before use
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const model = 'gemini-3-flash-preview';
  const prompt = `Break the task "${taskTitle}" into 3-5 simple, actionable micro-steps. Detect the input language and respond in the same language. Keep steps concise.`;
  
  const responseSchema = {
    type: Type.ARRAY,
    items: { type: Type.STRING }
  };

  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config: {
      systemInstruction: "You are a productivity coach. Your goal is to make complex tasks feel small and achievable. Detect the language of the input and respond exclusively in that language. If Hebrew, use masculine forms.",
      responseMimeType: "application/json",
      responseSchema
    }
  });

  const text = (response.text || "").trim();
  return JSON.parse(text);
};

export const generateSchedule = async (tasks: Task[]): Promise<ScheduleResult> => {
  // Initialize AI client right before use
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const model = 'gemini-3-pro-preview';
  const taskSummary = tasks.map(t => `ID: ${t.id}, Title: ${t.title}, Deadline: ${t.deadline}, Priority: ${t.priority}, Notes: ${t.notes || 'None'}`).join('\n');
  
  const prompt = `
Current Tasks:
${taskSummary}

Task: Organize these tasks into an optimized schedule. Detect the primary language used in the tasks and respond entirely in that language.
Consider:
1. Urgency (Deadlines).
2. Importance (Priority).
3. Notes/Context provided.
4. Logic (What should be done first to build momentum?).

Provide an order for each task ID and a short 'Reasoning' for why it's placed there.
`;

  const responseSchema = {
    type: Type.OBJECT,
    properties: {
      tasks: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            taskId: { type: Type.STRING },
            order: { type: Type.NUMBER },
            reasoning: { type: Type.STRING }
          },
          required: ["taskId", "order", "reasoning"]
        }
      },
      summary: { type: Type.STRING, description: "A one-sentence overall strategy for the day." }
    },
    required: ["tasks", "summary"]
  };

  const response = await ai.models.generateContent({
    model,
    contents: prompt,
    config: {
      systemInstruction: "You are an elite executive assistant. You specialize in ruthless prioritization and focus. Always detect the language of the task list and respond in that language. If Hebrew, use masculine forms.",
      responseMimeType: "application/json",
      responseSchema
    }
  });

  const text = (response.text || "").trim();
  return JSON.parse(text);
};
