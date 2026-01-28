
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { GoogleGenAI, LiveServerMessage, Modality, Blob } from '@google/genai';
import { Decision, WeightedItem, DecisionResult, Task, SubTask } from './types';
import { getDecisionRecommendation, breakDownTask, generateSchedule } from './geminiService';

// Audio Encoding/Decoding Utilities
function encode(bytes: Uint8Array) {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createBlob(data: Float32Array): Blob {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}

const Logo: React.FC<{ className?: string }> = ({ className = "" }) => (
  <div className={`flex items-center gap-2 sm:gap-4 md:gap-6 ${className}`}>
    <div className="flex items-center justify-center shrink-0">
      <svg width="24" height="24" className="sm:w-8 sm:h-8 md:w-11 md:h-11" viewBox="0 0 24 24" fill="none" stroke="#549090" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="11" r="5" />
        <path d="M12 3v2" />
        <path d="M19 8l-1.5 1" />
        <path d="M22 12h-2" />
        <path d="M19 16l-1.5-1" />
        <path d="M5 16l1.5-1" />
        <path d="M2 12h2" />
        <path d="M5 8l1.5 1" />
        <path d="M10 16c.3 1 1 1.5 2 1.5s1.7-.5 2-1.5" />
        <path d="M10 18h4" />
        <path d="M11 20h2" />
      </svg>
    </div>
    <div className="text-[#549090] text-sm xs:text-base sm:text-2xl md:text-4xl font-light tracking-[0.15em] sm:tracking-[0.25em] whitespace-nowrap pt-0.5">
      DECISION HELPER
    </div>
  </div>
);

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'Standard' | 'Focus' | 'Calendar' | 'History'>('Standard');
  const [standardSubMode, setStandardSubMode] = useState<'Quick' | 'Deep'>('Quick');
  const [dilemma, setDilemma] = useState('');
  const [prosText, setProsText] = useState('');
  const [consText, setConsText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DecisionResult | null>(null);
  const [history, setHistory] = useState<Decision[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Focus Mode State
  const [tasks, setTasks] = useState<Task[]>([]);
  const [newTaskTitle, setNewTaskTitle] = useState('');
  const [newTaskDeadline, setNewTaskDeadline] = useState('');
  const [newTaskPriority, setNewTaskPriority] = useState<'High' | 'Medium' | 'Low'>('Medium');
  const [newTaskNotes, setNewTaskNotes] = useState('');
  const [isScheduling, setIsScheduling] = useState(false);
  const [scheduleSummary, setScheduleSummary] = useState('');
  const [showScheduleLog, setShowScheduleLog] = useState(false);

  // Calendar Logic State
  const [currentDate, setCurrentDate] = useState(new Date());
  const [selectedCalendarDate, setSelectedCalendarDate] = useState<string>(new Date().toISOString().split('T')[0]);

  // Live Conversation State
  const [isListening, setIsListening] = useState(false);
  const [liveStatus, setLiveStatus] = useState<'IDLE' | 'LISTENING' | 'THINKING' | 'SPEAKING'>('IDLE');
  
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const streamRef = useRef<MediaStream | null>(null);
  const sessionRef = useRef<any>(null);

  useEffect(() => {
    const savedHistory = localStorage.getItem('decision_helper_history');
    if (savedHistory) setHistory(JSON.parse(savedHistory));

    const savedTasks = localStorage.getItem('decision_helper_tasks');
    if (savedTasks) setTasks(JSON.parse(savedTasks));
  }, []);

  useEffect(() => {
    localStorage.setItem('decision_helper_tasks', JSON.stringify(tasks));
  }, [tasks]);

  const saveToHistory = (newDecision: Decision) => {
    const updated = [newDecision, ...history].slice(0, 50);
    setHistory(updated);
    localStorage.setItem('decision_helper_history', JSON.stringify(updated));
  };

  const updateOutcome = (id: string, outcome: Decision['outcome']) => {
    const updated = history.map(d => d.id === id ? { ...d, outcome } : d);
    setHistory(updated);
    localStorage.setItem('decision_helper_history', JSON.stringify(updated));
  };

  const handleDecide = async () => {
    if (!dilemma.trim()) {
      setError("Tell me a bit about what's on your mind.");
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);

    const pros: WeightedItem[] = standardSubMode === 'Deep' 
      ? prosText.split('\n').filter(l => l.trim()).map(l => ({ id: uuidv4(), text: l.trim(), weight: 5 }))
      : [];
    const cons: WeightedItem[] = standardSubMode === 'Deep'
      ? consText.split('\n').filter(l => l.trim()).map(l => ({ id: uuidv4(), text: l.trim(), weight: 5 }))
      : [];

    try {
      const decisionData: Decision = {
        id: uuidv4(),
        dilemma,
        pros,
        cons,
        mode: standardSubMode === 'Deep' ? 'Standard' : 'Quick',
        createdAt: Date.now(),
        outcome: 'Pending'
      };

      const aiResponse = await getDecisionRecommendation(decisionData);
      setResult(aiResponse);
      saveToHistory({ 
        ...decisionData, 
        recommendation: aiResponse.recommendation, 
        explanation: aiResponse.explanation,
        confidence: aiResponse.confidence 
      });
    } catch (err: any) {
      setError(err.message || "I had some trouble thinking that through.");
    } finally {
      setLoading(false);
    }
  };

  // Focus Mode Logic
  const addTask = () => {
    if (!newTaskTitle.trim()) return;
    const newTask: Task = {
      id: uuidv4(),
      title: newTaskTitle,
      deadline: newTaskDeadline || new Date().toISOString().split('T')[0],
      priority: newTaskPriority,
      notes: newTaskNotes,
      completed: false,
      subTasks: []
    };
    setTasks([...tasks, newTask]);
    setNewTaskTitle('');
    setNewTaskDeadline('');
    setNewTaskNotes('');
  };

  const deleteTask = (id: string) => {
    setTasks(tasks.filter(t => t.id !== id));
  };

  const toggleTask = (id: string) => {
    setTasks(tasks.map(t => t.id === id ? { ...t, completed: !t.completed } : t));
  };

  const handleBreakDown = async (taskId: string) => {
    const task = tasks.find(t => t.id === taskId);
    if (!task) return;
    setLoading(true);
    try {
      const steps = await breakDownTask(task.title);
      const subTasks: SubTask[] = steps.map(s => ({ id: uuidv4(), text: s, completed: false }));
      setTasks(tasks.map(t => t.id === taskId ? { ...t, subTasks } : t));
    } catch (err) {
      setError("Failed to break down task.");
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateSchedule = async () => {
    const activeTasks = tasks.filter(t => !t.completed);
    if (activeTasks.length === 0) return;
    setIsScheduling(true);
    try {
      const schedule = await generateSchedule(activeTasks);
      setScheduleSummary(schedule.summary);
      setTasks(tasks.map(t => {
        const suggestion = schedule.tasks.find(s => s.taskId === t.id);
        if (suggestion) {
          return { ...t, suggestedOrder: suggestion.order, reasoning: suggestion.reasoning };
        }
        return t;
      }));
      setShowScheduleLog(true);
    } catch (err) {
      setError("Failed to generate schedule.");
    } finally {
      setIsScheduling(false);
    }
  };

  const resetForm = () => {
    setDilemma('');
    setProsText('');
    setConsText('');
    setResult(null);
    setError(null);
  };

  const stopListening = () => {
    setIsListening(false);
    setLiveStatus('IDLE');
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (inputAudioContextRef.current) {
      inputAudioContextRef.current.close();
      inputAudioContextRef.current = null;
    }
    if (outputAudioContextRef.current) {
      outputAudioContextRef.current.close();
      outputAudioContextRef.current = null;
    }
    for (const source of sourcesRef.current) {
      try { source.stop(); } catch(e) {}
    }
    sourcesRef.current.clear();
    nextStartTimeRef.current = 0;
    if (sessionRef.current) {
      sessionRef.current.then((session: any) => session.close());
      sessionRef.current = null;
    }
  };

  const toggleVoiceInput = async () => {
    if (isListening) {
      stopListening();
      return;
    }

    try {
      setIsListening(true);
      setLiveStatus('LISTENING');
      setError(null);
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
      
      const inAudioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      const outAudioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      inputAudioContextRef.current = inAudioCtx;
      outputAudioContextRef.current = outAudioCtx;

      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-12-2025',
        callbacks: {
          onopen: () => {
            const source = inAudioCtx.createMediaStreamSource(stream);
            const scriptProcessor = inAudioCtx.createScriptProcessor(4096, 1, 1);
            scriptProcessor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              const pcmBlob = createBlob(inputData);
              sessionPromise.then((session) => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };
            source.connect(scriptProcessor);
            scriptProcessor.connect(inAudioCtx.destination);
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.serverContent?.modelTurn) {
              setLiveStatus('SPEAKING');
            } else if (message.serverContent?.turnComplete) {
              setLiveStatus('LISTENING');
            }

            const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (base64Audio) {
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outAudioCtx.currentTime);
              const audioBuffer = await decodeAudioData(decode(base64Audio), outAudioCtx, 24000, 1);
              const source = outAudioCtx.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(outAudioCtx.destination);
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += audioBuffer.duration;
              sourcesRef.current.add(source);
              source.onended = () => {
                 sourcesRef.current.delete(source);
                 if (sourcesRef.current.size === 0) setLiveStatus('LISTENING');
              };
            }

            if (message.serverContent?.interrupted) {
              for (const s of sourcesRef.current) {
                try { s.stop(); } catch(e) {}
              }
              sourcesRef.current.clear();
              nextStartTimeRef.current = 0;
              setLiveStatus('LISTENING');
            }
          },
          onerror: (e) => {
            console.error("Live API Error:", e);
            stopListening();
          },
          onclose: () => {
            setIsListening(false);
            setLiveStatus('IDLE');
          },
        },
        config: {
          responseModalities: [Modality.AUDIO],
          inputAudioTranscription: {},
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } },
          },
          systemInstruction: `You are 'Decision Helper', a live voice-only advisor. Always speak using masculine forms (בלשון זכר) in Hebrew. Respond naturally in the language used by the user (Hebrew or English).`
        }
      });
      sessionRef.current = sessionPromise;

    } catch (err) {
      console.error("Voice error:", err);
      setIsListening(false);
      setLiveStatus('IDLE');
      setError("Microphone access is needed.");
    }
  };

  const getStatusColor = () => {
    switch(liveStatus) {
      case 'LISTENING': return 'bg-emerald-500';
      case 'SPEAKING': return 'bg-indigo-500';
      case 'THINKING': return 'bg-amber-500';
      default: return 'bg-rose-500';
    }
  };

  // Calendar Helpers
  const getDaysInMonth = (year: number, month: number) => new Date(year, month + 1, 0).getDate();
  const getFirstDayOfMonth = (year: number, month: number) => new Date(year, month, 1).getDay();

  const calendarGrid = useMemo(() => {
    const year = currentDate.getFullYear();
    const month = currentDate.getMonth();
    const daysInMonth = getDaysInMonth(year, month);
    const firstDay = getFirstDayOfMonth(year, month);
    
    const days = [];
    // Previous month padding
    const prevMonthDays = getDaysInMonth(year, month - 1);
    for (let i = firstDay - 1; i >= 0; i--) {
      days.push({ day: prevMonthDays - i, month: month - 1, year, isPadding: true });
    }
    // Current month
    for (let i = 1; i <= daysInMonth; i++) {
      days.push({ day: i, month, year, isPadding: false });
    }
    // Next month padding
    const remaining = 42 - days.length;
    for (let i = 1; i <= remaining; i++) {
      days.push({ day: i, month: month + 1, year, isPadding: true });
    }
    return days;
  }, [currentDate]);

  const tasksByDate = useMemo(() => {
    const map: Record<string, Task[]> = {};
    tasks.forEach(task => {
      const date = task.deadline || 'No Date';
      if (!map[date]) map[date] = [];
      map[date].push(task);
    });
    return map;
  }, [tasks]);

  const selectedDateTasks = useMemo(() => tasksByDate[selectedCalendarDate] || [], [tasksByDate, selectedCalendarDate]);

  const changeMonth = (offset: number) => {
    setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() + offset, 1));
  };

  const scheduledTasks = useMemo(() => tasks
    .filter(t => t.suggestedOrder !== undefined && !t.completed)
    .sort((a, b) => (a.suggestedOrder || 0) - (b.suggestedOrder || 0)), [tasks]);

  return (
    <div className="min-h-screen pb-40 selection:bg-indigo-100 relative overflow-x-hidden">
      <nav className="bg-white/95 backdrop-blur-lg sticky top-0 z-50 border-b border-slate-100 shadow-sm">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 h-16 sm:h-24 flex items-center justify-between gap-2">
          <Logo className="flex-shrink-0" />
          <div className="flex bg-slate-100/80 p-1 rounded-lg sm:rounded-2xl overflow-x-auto no-scrollbar">
            {(['Standard', 'Focus', 'Calendar', 'History'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => { setActiveTab(tab); setResult(null); setError(null); }}
                className={`whitespace-nowrap px-2.5 sm:px-5 py-1.5 sm:py-2.5 text-[10px] sm:text-xs font-bold rounded-lg sm:rounded-xl transition-all duration-300 ${activeTab === tab ? 'bg-white shadow-md text-indigo-600' : 'text-slate-500 hover:text-slate-800'}`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-3xl mx-auto px-4 sm:px-6 mt-6 sm:mt-12">
        {activeTab === 'Standard' && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-6 duration-700">
            <header className="text-center space-y-2 max-w-xl mx-auto">
              <h2 className="text-3xl sm:text-5xl font-extrabold text-black tracking-tight leading-tight">What's the dilemma?</h2>
              <div className="flex items-center justify-center gap-2 pt-1">
                 <button onClick={() => setStandardSubMode('Quick')} className={`px-4 py-1 rounded-full text-[10px] font-black uppercase tracking-widest ${standardSubMode === 'Quick' ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-400'}`}>Quick</button>
                 <button onClick={() => setStandardSubMode('Deep')} className={`px-4 py-1 rounded-full text-[10px] font-black uppercase tracking-widest ${standardSubMode === 'Deep' ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-400'}`}>In-depth</button>
              </div>
            </header>

            <div className="bg-white p-6 sm:p-12 rounded-[2rem] sm:rounded-[3.5rem] shadow-2xl border border-slate-50 space-y-6">
              <textarea
                value={dilemma}
                onChange={(e) => setDilemma(e.target.value)}
                placeholder="Should I... because..."
                dir="auto"
                className="w-full h-32 sm:h-44 p-6 sm:p-8 text-lg sm:text-xl bg-slate-50 rounded-[1.5rem] focus:bg-white focus:border-indigo-400 focus:outline-none transition-all resize-none font-medium text-black"
              />

              {standardSubMode === 'Deep' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <textarea
                    value={prosText}
                    onChange={(e) => setProsText(e.target.value)}
                    placeholder="Advantages..."
                    className="w-full h-32 p-5 bg-emerald-50/50 rounded-[1.5rem] focus:bg-white outline-none resize-none text-black"
                  />
                  <textarea
                    value={consText}
                    onChange={(e) => setConsText(e.target.value)}
                    placeholder="Drawbacks..."
                    className="w-full h-32 p-5 bg-rose-50/50 rounded-[1.5rem] focus:bg-white outline-none resize-none text-black"
                  />
                </div>
              )}

              <div className="flex flex-col items-center gap-6 pt-4">
                <button
                  onClick={handleDecide}
                  disabled={loading}
                  className="px-10 py-4 bg-slate-900 text-white rounded-full font-black text-lg shadow-xl hover:scale-105 transition-all disabled:opacity-50"
                >
                  {loading ? "Deciding..." : "Get Direction"}
                </button>
              </div>
            </div>

            {result && (
              <div className="bg-white p-8 rounded-[2.5rem] shadow-3xl border border-indigo-50 animate-in fade-in slide-in-from-bottom-6 duration-700">
                <div className="space-y-6" dir="auto">
                  <div className="flex justify-between items-center">
                    <span className="px-4 py-1 bg-indigo-50 text-indigo-600 rounded-full text-[10px] font-black uppercase tracking-widest">Recommendation</span>
                    <span className="text-xl font-black text-indigo-500">{result.confidence}% Match</span>
                  </div>
                  <h3 className="text-2xl font-extrabold text-black leading-tight">{result.recommendation}</h3>
                  <p className="text-slate-600 italic border-l-4 border-indigo-100 pl-4">{result.explanation}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'Focus' && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-6 duration-700">
            <header className="text-center space-y-2">
              <h2 className="text-3xl font-extrabold text-black">Focus Mode</h2>
              <p className="text-slate-500 font-medium">Prioritize your goals and comments with AI.</p>
            </header>

            <div className="bg-white p-6 rounded-[2rem] shadow-xl border border-slate-50 space-y-4">
              <input 
                type="text" 
                value={newTaskTitle}
                onChange={(e) => setNewTaskTitle(e.target.value)}
                placeholder="What's the target?" 
                dir="auto"
                className="w-full px-6 py-4 bg-slate-50 rounded-xl outline-none font-bold text-black placeholder:text-slate-400 border-2 border-transparent focus:border-indigo-200"
              />
              <div className="flex flex-wrap gap-4">
                <div className="flex-1 min-w-[140px] space-y-1">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Deadline</label>
                  <input 
                    type="date" 
                    value={newTaskDeadline} 
                    onChange={(e) => setNewTaskDeadline(e.target.value)} 
                    className="w-full px-4 py-3 bg-slate-50 rounded-xl font-bold text-black outline-none border-2 border-transparent focus:border-indigo-100" 
                  />
                </div>
                <div className="flex-1 min-w-[140px] space-y-1">
                  <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Priority</label>
                  <select 
                    value={newTaskPriority}
                    onChange={(e) => setNewTaskPriority(e.target.value as any)}
                    className="w-full px-4 py-3 bg-slate-50 rounded-xl font-bold text-black outline-none appearance-none border-2 border-transparent focus:border-indigo-100 cursor-pointer"
                  >
                    <option value="High">High Priority</option>
                    <option value="Medium">Medium Priority</option>
                    <option value="Low">Low Priority</option>
                  </select>
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest ml-2">Notes & Comments</label>
                <textarea 
                  value={newTaskNotes}
                  onChange={(e) => setNewTaskNotes(e.target.value)}
                  placeholder="Context, details, or obstacles..."
                  dir="auto"
                  className="w-full px-6 py-4 bg-slate-50 rounded-xl outline-none font-medium text-black h-28 resize-none placeholder:text-slate-400 border-2 border-transparent focus:border-indigo-100"
                />
              </div>
              <div className="flex gap-4">
                 <button onClick={addTask} className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white py-4 rounded-xl font-black transition-colors shadow-lg active:scale-95">Add Target</button>
              </div>
              
              {tasks.length > 0 && (
                <button 
                  onClick={handleGenerateSchedule}
                  disabled={isScheduling} 
                  className="w-full py-5 bg-slate-900 text-white rounded-full font-black text-sm shadow-xl hover:bg-black transition-all disabled:opacity-50 flex items-center justify-center gap-3"
                >
                   {isScheduling ? (
                     <>
                       <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                       Optimizing...
                     </>
                   ) : "✨ Optimize Daily Schedule"}
                </button>
              )}
            </div>

             {/* AI Optimized Schedule Log */}
             {showScheduleLog && scheduledTasks.length > 0 && (
              <div className="bg-slate-900 p-8 rounded-[2.5rem] shadow-2xl text-white animate-in slide-in-from-top-4 duration-500 relative overflow-hidden">
                <div className="absolute top-0 right-0 p-8 opacity-5 pointer-events-none">
                  <svg className="w-40 h-40" fill="currentColor" viewBox="0 0 24 24"><path d="M19 3h-1V1h-2v2H8V1H6v2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V8h14v11zM7 10h5v5H7z"/></svg>
                </div>
                
                <div className="relative z-10 space-y-6" dir="auto">
                  <div className="flex justify-between items-start">
                    <div className="space-y-1">
                       <h3 className="text-2xl font-black uppercase tracking-[0.2em] text-indigo-400">Tactical Strategy</h3>
                       {scheduleSummary && <p className="text-slate-300 text-lg font-medium italic">"{scheduleSummary}"</p>}
                    </div>
                    <button onClick={() => setShowScheduleLog(false)} className="text-slate-500 hover:text-white transition-colors">
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
                    </button>
                  </div>

                  <div className="space-y-6 relative pl-5 border-l-2 border-indigo-500/30">
                    {scheduledTasks.map((task) => (
                      <div key={task.id} className="relative">
                        <div className="absolute -left-[27px] top-1.5 w-3.5 h-3.5 rounded-full bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]" />
                        <div className="space-y-1.5">
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] font-black bg-indigo-500/20 text-indigo-300 px-2 py-0.5 rounded uppercase tracking-widest">Step {task.suggestedOrder}</span>
                            <span className={`text-[10px] font-black px-2 py-0.5 rounded uppercase tracking-widest ${task.priority === 'High' ? 'bg-rose-500/20 text-rose-400' : task.priority === 'Medium' ? 'bg-amber-500/20 text-amber-400' : 'bg-slate-700 text-slate-400'}`}>{task.priority} Priority</span>
                          </div>
                          <h4 className="text-xl font-bold">{task.title}</h4>
                          {task.reasoning && (
                            <p className="text-sm text-slate-400 font-medium leading-relaxed max-w-xl">{task.reasoning}</p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <div className="space-y-4">
              {tasks.filter(t => !t.completed).map(task => (
                <div key={task.id} className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 transition-all hover:border-indigo-100 group">
                  <div className="flex justify-between items-start">
                    <div className="flex gap-4 items-start">
                      <button onClick={() => toggleTask(task.id)} className="w-7 h-7 rounded-full border-2 border-slate-200 mt-1 hover:border-indigo-400 hover:bg-indigo-50 transition-all flex items-center justify-center group-hover:scale-110" />
                      <div className="space-y-1">
                        <div className="flex flex-wrap gap-2 items-center">
                          <span className={`text-[9px] font-black px-2 py-0.5 rounded uppercase tracking-widest ${task.priority === 'High' ? 'bg-rose-100 text-rose-600' : task.priority === 'Medium' ? 'bg-amber-100 text-amber-600' : 'bg-slate-100 text-slate-400'}`}>
                             {task.priority} Priority
                          </span>
                          <span className="text-[10px] text-slate-400 uppercase font-black">{task.deadline}</span>
                          {task.suggestedOrder && <span className="text-[9px] font-black bg-indigo-50 text-indigo-600 px-2 py-0.5 rounded uppercase tracking-widest">Planned: Step {task.suggestedOrder}</span>}
                        </div>
                        <h4 className="text-xl font-extrabold text-black">{task.title}</h4>
                        {task.notes && (
                          <div className="bg-slate-50/50 p-4 rounded-xl mt-3 border-l-2 border-indigo-200">
                            <p className="text-sm text-slate-700 font-medium" dir="auto">{task.notes}</p>
                          </div>
                        )}
                        {task.reasoning && (
                          <div className="flex items-center gap-2 pt-2">
                             <div className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
                             <p className="text-xs text-indigo-600 font-bold italic" dir="auto">AI Strategy: {task.reasoning}</p>
                          </div>
                        )}
                      </div>
                    </div>
                    <button onClick={() => deleteTask(task.id)} className="text-slate-300 hover:text-rose-500 transition-colors p-2">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M6 18L18 6M6 6l12 12" /></svg>
                    </button>
                  </div>
                  
                  {/* Breakdown Option */}
                  {task.subTasks.length === 0 && (
                     <div className="pl-11 mt-4">
                        <button onClick={() => handleBreakDown(task.id)} className="text-[10px] font-black text-indigo-500 hover:text-indigo-700 hover:underline uppercase tracking-widest flex items-center gap-1">
                           + Breakdown to Action Steps
                        </button>
                     </div>
                  )}
                  {/* Subtasks Display */}
                  {task.subTasks.length > 0 && (
                    <div className="pl-11 mt-4 space-y-2 border-l-2 border-indigo-50 ml-3.5">
                      {task.subTasks.map(st => (
                         <div key={st.id} className="flex items-center gap-2 text-sm text-slate-800 font-bold">
                            <div className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
                            {st.text}
                         </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {tasks.filter(t => !t.completed).length === 0 && (
                 <div className="text-center py-20 bg-slate-50/50 rounded-[2rem] border-2 border-dashed border-slate-100">
                    <p className="text-slate-400 font-bold">Your focus targets are empty.</p>
                 </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'Calendar' && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-6 duration-700">
            <header className="flex items-center justify-between">
              <div>
                <h2 className="text-3xl font-extrabold text-black">
                  {currentDate.toLocaleDateString('he-IL', { month: 'long', year: 'numeric' })}
                </h2>
                <p className="text-slate-500 text-sm font-bold uppercase tracking-widest mt-1">לוח תכנון אישי</p>
              </div>
              <div className="flex gap-2">
                <button onClick={() => changeMonth(-1)} className="p-2 hover:bg-slate-100 rounded-full transition-colors text-slate-600">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" /></svg>
                </button>
                <button onClick={() => setCurrentDate(new Date())} className="px-4 py-2 bg-slate-100 text-slate-600 rounded-full text-xs font-black uppercase">היום</button>
                <button onClick={() => changeMonth(1)} className="p-2 hover:bg-slate-100 rounded-full transition-colors text-slate-600">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" /></svg>
                </button>
              </div>
            </header>

            <div className="bg-white rounded-[2.5rem] p-6 shadow-2xl shadow-indigo-100/40 border border-slate-50">
              <div className="grid grid-cols-7 mb-4 text-center">
                {['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ש'].map(day => (
                  <div key={day} className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{day}</div>
                ))}
              </div>

              <div className="grid grid-cols-7 gap-1 sm:gap-2">
                {calendarGrid.map((item, idx) => {
                  const dateStr = `${item.year}-${String(item.month + 1).padStart(2, '0')}-${String(item.day).padStart(2, '0')}`;
                  const isSelected = selectedCalendarDate === dateStr;
                  const isToday = new Date().toISOString().split('T')[0] === dateStr;
                  const dayTasks = tasksByDate[dateStr] || [];
                  
                  return (
                    <button
                      key={idx}
                      onClick={() => setSelectedCalendarDate(dateStr)}
                      className={`relative aspect-square rounded-2xl flex flex-col items-center justify-center transition-all duration-300 group
                        ${item.isPadding ? 'opacity-20 grayscale' : 'hover:bg-indigo-50'}
                        ${isSelected ? 'bg-indigo-600 text-white shadow-lg scale-105' : 'bg-transparent'}
                        ${isToday && !isSelected ? 'border-2 border-indigo-200' : ''}
                      `}
                    >
                      <span className={`text-sm sm:text-lg font-bold ${isSelected ? 'text-white' : 'text-slate-700'}`}>{item.day}</span>
                      
                      <div className="flex gap-0.5 mt-1">
                        {dayTasks.slice(0, 3).map((t, i) => (
                          <div 
                            key={i} 
                            className={`w-1 h-1 rounded-full ${
                              isSelected ? 'bg-white/50' : 
                              t.priority === 'High' ? 'bg-rose-400' : 
                              t.priority === 'Medium' ? 'bg-amber-400' : 'bg-indigo-400'
                            }`} 
                          />
                        ))}
                        {dayTasks.length > 3 && <div className={`w-1 h-1 rounded-full ${isSelected ? 'bg-white/50' : 'bg-slate-300'}`} />}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="animate-in slide-in-from-top-4 duration-500">
              <div className="flex items-center justify-between px-4 mb-4">
                <h3 className="text-xl font-black text-black">
                  {new Date(selectedCalendarDate).toLocaleDateString('he-IL', { weekday: 'long', day: 'numeric', month: 'long' })}
                </h3>
                <span className="text-[10px] font-black text-indigo-400 bg-indigo-50 px-3 py-1 rounded-full uppercase tracking-widest">
                  {selectedDateTasks.length} משימות
                </span>
              </div>

              {selectedDateTasks.length === 0 ? (
                <div className="text-center py-12 bg-slate-50/50 rounded-[2rem] border-2 border-dashed border-slate-100">
                  <p className="text-slate-400 font-bold">אין משימות מתוכננות ליום זה</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {selectedDateTasks.map(task => (
                    <div key={task.id} className={`bg-white p-5 rounded-2xl shadow-sm border border-slate-50 flex items-center justify-between transition-all ${task.completed ? 'opacity-40 grayscale' : ''}`}>
                      <div className="flex items-center gap-4">
                        <button onClick={() => toggleTask(task.id)} className={`w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all ${task.completed ? 'bg-emerald-500 border-emerald-500' : 'border-slate-200'}`}>
                          {task.completed && <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7" /></svg>}
                        </button>
                        <div className="space-y-1">
                          <h4 className={`font-bold text-black ${task.completed ? 'line-through' : ''}`}>{task.title}</h4>
                          <div className="flex gap-2">
                             <span className={`text-[8px] font-black px-1.5 py-0.5 rounded uppercase ${task.priority === 'High' ? 'bg-rose-100 text-rose-500' : task.priority === 'Medium' ? 'bg-amber-100 text-amber-500' : 'bg-slate-100 text-slate-400'}`}>
                                {task.priority} Priority
                             </span>
                             {task.notes && <span className="text-[8px] font-bold text-slate-400 uppercase tracking-widest">• Note attached</span>}
                          </div>
                        </div>
                      </div>
                      <button onClick={() => deleteTask(task.id)} className="text-slate-200 hover:text-rose-400 transition-colors">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'History' && (
          <div className="space-y-8 animate-in fade-in duration-700">
            <h2 className="text-3xl font-extrabold text-black">History</h2>
            {history.length === 0 ? (
               <div className="text-center py-20 bg-white rounded-[2rem] border-2 border-dashed border-slate-100">
                  <p className="text-slate-400 font-bold">Your decision logs are empty.</p>
               </div>
            ) : (
              <div className="grid gap-4">
                {history.map(item => (
                  <div key={item.id} className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm transition-shadow hover:shadow-md">
                    <div className="flex justify-between items-start mb-2">
                       <h4 className="font-extrabold text-black text-lg" dir="auto">{item.dilemma}</h4>
                       <span className="text-[10px] text-slate-400 font-black">{new Date(item.createdAt).toLocaleDateString()}</span>
                    </div>
                    <p className="text-sm text-indigo-600 font-bold border-t border-slate-50 pt-3 mt-3">{item.recommendation}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Voice Assistant Toggle */}
      <div className="fixed bottom-10 right-10 z-50">
        <button
          onClick={toggleVoiceInput}
          className={`w-20 h-20 rounded-full flex items-center justify-center shadow-3xl transition-all duration-500 ${isListening ? 'bg-rose-500 scale-110' : 'bg-slate-900 hover:bg-black'}`}
        >
          <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default App;
