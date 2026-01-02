
import React, { useState, useEffect, useRef } from 'react';
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
    {/* Refined Minimalist Lightbulb Icon in Teal #549090 */}
    <div className="flex items-center justify-center shrink-0">
      <svg width="24" height="24" className="sm:w-8 sm:h-8 md:w-11 md:h-11" viewBox="0 0 24 24" fill="none" stroke="#549090" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
        {/* Main bulb circle */}
        <circle cx="12" cy="11" r="5" />
        {/* Radiating segments matching image layout */}
        <path d="M12 3v2" /> {/* Top */}
        <path d="M19 8l-1.5 1" /> {/* Top Right */}
        <path d="M22 12h-2" /> {/* Right */}
        <path d="M19 16l-1.5-1" /> {/* Bottom Right */}
        <path d="M5 16l1.5-1" /> {/* Bottom Left */}
        <path d="M2 12h2" /> {/* Left */}
        <path d="M5 8l1.5 1" /> {/* Top Left */}
        {/* Bulb base detail */}
        <path d="M10 16c.3 1 1 1.5 2 1.5s1.7-.5 2-1.5" />
        <path d="M10 18h4" />
        <path d="M11 20h2" />
      </svg>
    </div>
    {/* Clean Modern Sans-Serif Text with High Tracking - Scaled for Mobile */}
    <div className="text-[#549090] text-sm xs:text-base sm:text-2xl md:text-4xl font-light tracking-[0.15em] sm:tracking-[0.25em] whitespace-nowrap pt-0.5">
      DECISION HELPER
    </div>
  </div>
);

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'Standard' | 'Focus' | 'History'>('Standard');
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
  const [isScheduling, setIsScheduling] = useState(false);
  const [scheduleSummary, setScheduleSummary] = useState('');

  // Live Conversation Refs & State
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
      deadline: newTaskDeadline || 'Today',
      priority: newTaskPriority,
      completed: false,
      subTasks: []
    };
    setTasks([...tasks, newTask]);
    setNewTaskTitle('');
    setNewTaskDeadline('');
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
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
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

  const sortedTasks = [...tasks].sort((a, b) => {
    if (a.completed !== b.completed) return a.completed ? 1 : -1;
    if (a.suggestedOrder !== undefined && b.suggestedOrder !== undefined) return a.suggestedOrder - b.suggestedOrder;
    return 0;
  });

  return (
    <div className="min-h-screen pb-40 selection:bg-indigo-100 relative overflow-x-hidden">
      <nav className="bg-white/95 backdrop-blur-lg sticky top-0 z-50 border-b border-slate-100 shadow-sm">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 h-16 sm:h-24 flex items-center justify-between gap-2">
          <Logo className="flex-shrink-0" />
          <div className="flex bg-slate-100/80 p-1 rounded-xl sm:rounded-2xl">
            {(['Standard', 'Focus', 'History'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => { setActiveTab(tab); setResult(null); setError(null); }}
                className={`px-2 sm:px-6 py-1.5 sm:py-2.5 text-[10px] sm:text-xs font-bold rounded-lg sm:rounded-xl transition-all duration-300 ${activeTab === tab ? 'bg-white shadow-md text-indigo-600' : 'text-slate-500 hover:text-slate-800'}`}
              >
                {tab === 'Standard' ? (window.innerWidth < 640 ? 'Start' : 'Dilemma') : tab}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-3xl mx-auto px-4 sm:px-6 mt-6 sm:mt-12">
        {activeTab === 'Standard' && (
          <div className="space-y-8 sm:y-12 animate-in fade-in slide-in-from-bottom-6 duration-700">
            <header className="text-center space-y-2 sm:space-y-4 max-w-xl mx-auto">
              <h2 className="text-3xl sm:text-5xl font-extrabold text-slate-900 tracking-tight leading-tight">What's the dilemma?</h2>
              <div className="flex items-center justify-center gap-2 pt-1 sm:pt-2">
                 <button 
                  onClick={() => setStandardSubMode('Quick')}
                  className={`px-3 sm:px-4 py-1 sm:py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest transition-all ${standardSubMode === 'Quick' ? 'bg-indigo-600 text-white shadow-lg' : 'bg-slate-100 text-slate-400 hover:text-slate-600'}`}
                 >
                   Quick
                 </button>
                 <button 
                  onClick={() => setStandardSubMode('Deep')}
                  className={`px-3 sm:px-4 py-1 sm:py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest transition-all ${standardSubMode === 'Deep' ? 'bg-indigo-600 text-white shadow-lg' : 'bg-slate-100 text-slate-400 hover:text-slate-600'}`}
                 >
                   In-depth
                 </button>
              </div>
            </header>

            <div className="bg-white p-6 sm:p-12 rounded-[2rem] sm:rounded-[3.5rem] shadow-2xl shadow-indigo-100/40 border border-slate-50 space-y-6 sm:space-y-10">
              <div className="space-y-4">
                <div className="flex justify-between items-center px-2">
                  <label className="block text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Dilemma Input</label>
                  {isListening && (
                    <div className="flex items-center gap-2">
                       <div className={`w-2 h-2 rounded-full animate-pulse ${getStatusColor()}`} />
                       <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Call active</span>
                    </div>
                  )}
                </div>
                <textarea
                  value={dilemma}
                  onChange={(e) => setDilemma(e.target.value)}
                  placeholder="Should I... because..."
                  dir="auto"
                  className={`w-full h-32 sm:h-44 p-6 sm:p-8 text-lg sm:text-xl bg-slate-50/50 border-2 border-transparent rounded-[1.5rem] sm:rounded-[2.5rem] focus:bg-white focus:border-indigo-400 focus:outline-none transition-all resize-none placeholder:text-slate-300 font-medium leading-relaxed ${error && !dilemma.trim() ? 'border-rose-300 bg-rose-50/20' : ''}`}
                />
              </div>

              {standardSubMode === 'Deep' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 sm:gap-10 animate-in slide-in-from-top-4 duration-500">
                  <div className="space-y-4">
                    <label className="block text-[10px] font-black text-emerald-600 uppercase tracking-[0.2em] ml-2">Advantages</label>
                    <textarea
                      value={prosText}
                      onChange={(e) => setProsText(e.target.value)}
                      placeholder="List key benefits..."
                      dir="auto"
                      className="w-full h-32 sm:h-44 p-5 sm:p-7 text-sm sm:text-base bg-pastel-emerald/60 border-2 border-transparent rounded-[1.5rem] sm:rounded-[2rem] focus:bg-white focus:border-emerald-300 focus:outline-none transition-all resize-none placeholder:text-emerald-300 font-medium"
                    />
                  </div>
                  <div className="space-y-4">
                    <label className="block text-[10px] font-black text-rose-600 uppercase tracking-[0.2em] ml-2">Risks / Cons</label>
                    <textarea
                      value={consText}
                      onChange={(e) => setConsText(e.target.value)}
                      placeholder="List key drawbacks..."
                      dir="auto"
                      className="w-full h-32 sm:h-44 p-5 sm:p-7 text-sm sm:text-base bg-pastel-rose/60 border-2 border-transparent rounded-[1.5rem] sm:rounded-[2rem] focus:bg-white focus:border-rose-300 focus:outline-none transition-all resize-none placeholder:text-rose-300 font-medium"
                    />
                  </div>
                </div>
              )}

              {error && <p className="text-rose-500 text-center text-sm font-bold animate-pulse">{error}</p>}

              <div className="flex flex-col items-center gap-6 sm:gap-8 pt-2 sm:pt-4">
                <button
                  onClick={handleDecide}
                  disabled={loading}
                  className="group relative w-full sm:w-auto px-10 sm:px-16 py-4 sm:py-6 bg-slate-900 text-white rounded-full font-black text-lg sm:text-xl shadow-xl hover:shadow-2xl transition-all duration-300 disabled:opacity-50"
                >
                  <div className="relative z-10 flex items-center justify-center gap-4">
                    {loading ? "Deciding..." : "Get Direction"}
                  </div>
                  <div className="absolute inset-0 bg-gradient-to-r from-indigo-700 to-indigo-50 opacity-0 group-hover:opacity-100 transition-opacity rounded-full" />
                </button>
                <button onClick={resetForm} className="text-slate-400 text-[10px] font-black hover:text-slate-600 transition-colors uppercase tracking-widest">Start Fresh</button>
              </div>
            </div>

            {result && (
              <div className="bg-white p-8 sm:p-12 rounded-[2.5rem] sm:rounded-[4rem] shadow-3xl border border-indigo-50 animate-in fade-in slide-in-from-bottom-12 duration-1000">
                <div className="space-y-6 sm:space-y-10" dir="auto">
                  <div className="flex justify-between items-center">
                    <span className="px-4 sm:px-6 py-1.5 sm:py-2 bg-indigo-50 text-indigo-600 rounded-full text-[10px] font-black uppercase tracking-[0.2em]">Our Path Forward</span>
                    <span className="text-xl sm:text-2xl font-black text-indigo-500">{result.confidence}%</span>
                  </div>
                  <div className="space-y-4 sm:space-y-6">
                    <h3 className="text-2xl sm:text-4xl font-extrabold text-slate-900 leading-[1.2]">{result.recommendation}</h3>
                    <p className="text-base sm:text-xl text-slate-600 italic border-l-4 border-indigo-100 pl-4 sm:pl-6">"{result.explanation}"</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'Focus' && (
          <div className="space-y-8 sm:y-10 animate-in fade-in slide-in-from-bottom-6 duration-700">
            <header className="text-center space-y-3 sm:space-y-4 max-w-xl mx-auto">
              <h2 className="text-3xl sm:text-5xl font-extrabold text-slate-900 tracking-tight">Focus Mode</h2>
              <p className="text-base sm:text-lg text-slate-500 font-medium">Turn overwhelming tasks into an AI-powered schedule.</p>
            </header>

            <div className="bg-white p-6 sm:p-8 rounded-[2rem] sm:rounded-[3rem] shadow-xl border border-slate-50 space-y-6">
              <div className="flex flex-col gap-4">
                <input 
                  type="text" 
                  value={newTaskTitle}
                  onChange={(e) => setNewTaskTitle(e.target.value)}
                  placeholder="What needs to be done?" 
                  dir="auto"
                  className="flex-1 px-5 py-3 sm:px-6 sm:py-4 bg-slate-50 rounded-xl sm:rounded-2xl border-2 border-transparent focus:border-indigo-300 outline-none font-bold text-slate-700 text-sm sm:text-base"
                />
                <div className="flex flex-wrap gap-2 sm:gap-4">
                  <input 
                    type="date" 
                    value={newTaskDeadline}
                    onChange={(e) => setNewTaskDeadline(e.target.value)}
                    className="flex-1 min-w-[120px] px-4 py-3 sm:px-6 sm:py-4 bg-slate-50 rounded-xl sm:rounded-2xl border-2 border-transparent focus:border-indigo-300 outline-none font-bold text-slate-700 text-sm sm:text-base"
                  />
                  <select 
                    value={newTaskPriority}
                    onChange={(e) => setNewTaskPriority(e.target.value as any)}
                    className="flex-1 min-w-[100px] px-4 py-3 sm:px-6 sm:py-4 bg-slate-50 rounded-xl sm:rounded-2xl border-2 border-transparent focus:border-indigo-300 outline-none font-bold text-slate-700 appearance-none text-sm sm:text-base"
                  >
                    <option value="High">High</option>
                    <option value="Medium">Medium</option>
                    <option value="Low">Low</option>
                  </select>
                  <button onClick={addTask} className="bg-indigo-600 text-white px-6 sm:px-8 py-3 sm:py-4 rounded-xl sm:rounded-2xl font-black hover:bg-indigo-700 transition-all">+</button>
                </div>
              </div>

              {tasks.length > 0 && (
                <div className="flex justify-center pt-2 sm:pt-4">
                  <button 
                    onClick={handleGenerateSchedule}
                    disabled={isScheduling}
                    className="w-full sm:w-auto bg-slate-900 text-white px-8 sm:px-10 py-3 sm:py-4 rounded-full font-black text-sm shadow-lg hover:scale-105 transition-all disabled:opacity-50"
                  >
                    {isScheduling ? "Generating Schedule..." : "✨ AI Focus Schedule"}
                  </button>
                </div>
              )}
            </div>

            {scheduleSummary && (
              <div className="bg-indigo-50 p-5 sm:p-6 rounded-[1.5rem] sm:rounded-[2rem] border border-indigo-100 animate-in fade-in" dir="auto">
                <p className="text-indigo-800 font-bold italic text-center text-sm sm:text-base">"{scheduleSummary}"</p>
              </div>
            )}

            <div className="space-y-6">
              {sortedTasks.map((task) => (
                <div key={task.id} className={`bg-white p-6 sm:p-8 rounded-[1.5rem] sm:rounded-[2.5rem] shadow-sm border-2 transition-all ${task.completed ? 'opacity-50 grayscale' : 'border-slate-50 hover:border-indigo-100'}`}>
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex gap-3 sm:gap-4 items-start">
                      <button onClick={() => toggleTask(task.id)} className={`w-6 h-6 sm:w-8 sm:h-8 rounded-full border-2 flex items-center justify-center transition-all ${task.completed ? 'bg-emerald-500 border-emerald-500' : 'border-slate-200'}`}>
                        {task.completed && <svg className="w-4 h-4 sm:w-5 sm:h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7" /></svg>}
                      </button>
                      <div className="space-y-1" dir="auto">
                        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
                          {task.suggestedOrder !== undefined && <span className="text-[9px] sm:text-[10px] font-black bg-indigo-100 text-indigo-600 px-2 py-0.5 rounded-md">STEP {task.suggestedOrder}</span>}
                          <span className={`text-[9px] sm:text-[10px] font-black px-2 py-0.5 rounded-md ${task.priority === 'High' ? 'bg-rose-100 text-rose-600' : task.priority === 'Medium' ? 'bg-amber-100 text-amber-600' : 'bg-slate-100 text-slate-500'}`}>{task.priority} Impact</span>
                          <span className="text-[9px] sm:text-[10px] font-black text-slate-400 uppercase tracking-widest">{task.deadline}</span>
                        </div>
                        <h4 className={`text-xl sm:text-2xl font-extrabold text-slate-900 ${task.completed ? 'line-through' : ''}`}>{task.title}</h4>
                      </div>
                    </div>
                    <button onClick={() => deleteTask(task.id)} className="text-slate-300 hover:text-rose-500 transition-colors">
                      <svg className="w-5 h-5 sm:w-6 sm:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
                    </button>
                  </div>

                  {task.reasoning && !task.completed && (
                    <p className="text-xs sm:text-sm font-bold text-indigo-500/80 mb-6 bg-indigo-50/50 p-3 sm:p-4 rounded-xl sm:rounded-2xl italic" dir="auto">{task.reasoning}</p>
                  )}

                  {!task.completed && (
                    <div className="pl-9 sm:pl-12 space-y-3 sm:space-y-4" dir="auto">
                      {task.subTasks.length > 0 ? (
                        <div className="space-y-2">
                          {task.subTasks.map(st => (
                            <div key={st.id} className="flex items-center gap-2 sm:gap-3 text-slate-600 font-bold text-xs sm:text-sm">
                              <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full" />
                              {st.text}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <button 
                          onClick={() => handleBreakDown(task.id)}
                          className="text-[9px] sm:text-[10px] font-black text-indigo-600 hover:text-indigo-800 uppercase tracking-widest border-b border-indigo-200"
                        >
                          Break it down into steps
                        </button>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'History' && (
          <div className="space-y-8 sm:y-10 animate-in fade-in duration-700">
            <h2 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">Choice History</h2>
            {history.length === 0 ? (
              <div className="text-center py-16 sm:py-24 bg-white rounded-[2rem] sm:rounded-[3rem] border-2 border-dashed border-slate-100">
                <p className="text-slate-400 font-bold text-base sm:text-lg">No decisions logged yet.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-4 sm:gap-6">
                {history.map(item => (
                  <div key={item.id} className="bg-white p-6 sm:p-8 rounded-[1.5rem] sm:rounded-[2.5rem] border border-slate-100 shadow-sm hover:shadow-md transition-shadow group">
                    <div className="flex justify-between items-start mb-4 sm:mb-6" dir="auto">
                      <div className="space-y-2">
                        <div className="flex items-center gap-3">
                           <span className="text-[9px] sm:text-[10px] font-black text-indigo-400 uppercase tracking-widest">{new Date(item.createdAt).toLocaleDateString()}</span>
                        </div>
                        <h4 className="text-lg sm:text-xl font-bold text-slate-800 line-clamp-2">{item.dilemma}</h4>
                      </div>
                      <select 
                        value={item.outcome} 
                        onChange={(e) => updateOutcome(item.id, e.target.value as any)}
                        className="text-[9px] sm:text-[10px] font-black px-3 py-1.5 sm:px-4 sm:py-2 rounded-full border-none focus:ring-2 focus:ring-indigo-100 bg-slate-100"
                      >
                        <option value="Pending">Waiting...</option>
                        <option value="Followed">Followed</option>
                        <option value="Ignored">Ignored</option>
                      </select>
                    </div>
                    <div className="bg-slate-50/50 p-4 sm:p-6 rounded-2xl sm:rounded-3xl" dir="auto">
                      <p className="text-sm sm:text-base font-bold text-slate-700">{item.recommendation}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Floating Microphone Button - Anchored Fixed at Bottom-Right */}
      {activeTab !== 'History' && (
        <div className="fixed bottom-4 right-4 sm:bottom-10 sm:right-10 z-[100]">
          <div className="relative">
             {isListening && <div className="absolute inset-0 rounded-full animate-ping bg-indigo-500 opacity-20 scale-125 md:scale-150" />}
             <button
              onClick={toggleVoiceInput}
              className={`relative w-14 h-14 sm:w-20 sm:h-20 rounded-full flex items-center justify-center shadow-3xl transition-all duration-500 ${
                isListening ? 'bg-rose-500 scale-110' : 'bg-slate-900 hover:bg-black hover:scale-105'
              }`}
            >
              <svg className="w-6 h-6 sm:w-10 sm:h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            </button>
          </div>
        </div>
      )}

      <footer className="mt-24 sm:mt-40 pb-16 flex flex-col items-center gap-6 text-slate-300 font-medium text-sm">
        <Logo className="opacity-15 grayscale scale-75 sm:scale-90" />
        <p className="text-[9px] sm:text-[10px] font-black uppercase tracking-[0.3em] opacity-30">© 2025 Decision Helper Live</p>
      </footer>
    </div>
  );
};

export default App;
