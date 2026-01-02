
export interface WeightedItem {
  id: string;
  text: string;
  weight: number; // 1-10
}

export interface Decision {
  id: string;
  dilemma: string;
  pros: WeightedItem[];
  cons: WeightedItem[];
  mode: 'Standard' | 'Quick';
  recommendation?: string;
  explanation?: string;
  confidence?: number;
  outcome?: 'Followed' | 'Ignored' | 'Pending';
  createdAt: number;
}

export interface DecisionResult {
  recommendation: string;
  explanation: string;
  confidence: number;
  advantages?: string[];
  disadvantages?: string[];
}

export interface SubTask {
  id: string;
  text: string;
  completed: boolean;
}

export interface Task {
  id: string;
  title: string;
  deadline: string;
  priority: 'High' | 'Medium' | 'Low';
  completed: boolean;
  subTasks: SubTask[];
  suggestedOrder?: number;
  reasoning?: string;
}

export interface ScheduleResult {
  tasks: { taskId: string; order: number; reasoning: string }[];
  summary: string;
}
