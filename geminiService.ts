
import { GoogleGenAI, Type } from "@google/genai";
import { Decision, DecisionResult, Task, ScheduleResult } from "./types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });

export const getDecisionRecommendation = async (decision: Decision): Promise<DecisionResult> => {
  const model = 'gemini-3-flash-preview';
  
  const dilemmaLength = decision.dilemma.length;
  const isLongInput = dilemmaLength > 200 || (decision.pros.length + decision.cons.length > 0);
  
  const prosText = decision.pros.length > 0 
    ? "User-provided Advantages:\n" + decision.pros.map(p => `- ${p.text}`).join('\n')
    : "";
  const consText = decision.cons.length > 0 
    ? "User-provided Disadvantages:\n" + decision.cons.map(c => `- ${c.text}`).join('\n')
    : "";

  const prompt = `
Dilemma: ${decision.dilemma}
${prosText}
${consText}

Task: Provide a decisive and empathetic recommendation.
${isLongInput 
  ? "Since this is a detailed dilemma, provide a structured response including a list of core advantages and disadvantages you've identified." 
  : "Since this is a short dilemma, be extremely concise and punchy."}

Detect the language of the dilemma and respond in that same language.
`;

  const responseSchema = {
    type: Type.OBJECT,
    properties: {
      recommendation: { type: Type.STRING, description: "A clear, decisive action-oriented recommendation." },
      explanation: { type: Type.STRING, description: "A human-like explanation of the reasoning." },
      confidence: { type: Type.NUMBER, description: "Confidence level 0-100." },
      advantages: { 
        type: Type.ARRAY, 
        items: { type: Type.STRING }, 
        description: "Key advantages to follow this choice (for detailed analysis)." 
      },
      disadvantages: { 
        type: Type.ARRAY, 
        items: { type: Type.STRING }, 
        description: "Key risks or disadvantages to consider (for detailed analysis)." 
      },
    },
    required: ["recommendation", "explanation", "confidence"]
  };

  try {
    const response = await ai.models.generateContent({
      model,
      contents: prompt,
      config: {
        systemInstruction: `You are 'Decision Helper'. You help users solve daily dilemmas. 
        Be friendly and empathetic, but prioritize being DECISIVE. 
        Always respond in the same language as the user's input. 
        In Hebrew, use masculine forms (בלשון זכר).`,
        responseMimeType: "application/json",
        responseSchema,
        temperature: 0.7,
      },
    });

    return JSON.parse(response.text);
  } catch (error) {
    console.error("Gemini API Error:", error);
    throw new Error("My thought process was interrupted. Could you try asking again?");
  }
};

export const breakDownTask = async (taskTitle: string): Promise<string[]> => {
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

  return JSON.parse(response.text);
};

export const generateSchedule = async (tasks: Task[]): Promise<ScheduleResult> => {
  const model = 'gemini-3-pro-preview';
  const taskSummary = tasks.map(t => `ID: ${t.id}, Title: ${t.title}, Deadline: ${t.deadline}, Priority: ${t.priority}`).join('\n');
  
  const prompt = `
Current Tasks:
${taskSummary}

Task: Organize these tasks into an optimized schedule. Detect the primary language used in the tasks and respond entirely in that language.
Consider:
1. Urgency (Deadlines).
2. Importance (Priority).
3. Logic (What should be done first to build momentum?).

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

  return JSON.parse(response.text);
};
