import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Calendar,
  Clock3,
  FileImage,
  History,
  ScanLine,
  ShieldCheck,
} from "lucide-react";
import { getAnalysisHistory } from "../services/api";

export default function Dashboard({ onAnalysis, onHistory }) {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const loadDashboard = async () => {
      try {
        setLoading(true);
        setError("");
        const data = await getAnalysisHistory();
        setAnalyses(data.analyses || []);
      } catch (err) {
        console.error("DASHBOARD API ERROR:", err);
        setError(
          err.response?.data?.detail ||
            "Unable to load dashboard data."
        );
      } finally {
        setLoading(false);
      }
    };

    loadDashboard();
  }, []);

  const stats = useMemo(() => {
    const totalDetected = analyses.reduce(
      (sum, analysis) =>
        sum + (analysis.detected_conditions?.length || 0),
      0
    );

    const conditionCounts = {};

    analyses.forEach((analysis) => {
      (analysis.detected_conditions || []).forEach((condition) => {
        if (!conditionCounts[condition.label]) {
          conditionCounts[condition.label] = {
            label: condition.label,
            name: condition.name,
            count: 0,
          };
        }
        conditionCounts[condition.label].count += 1;
      });
    });

    return {
      totalAnalyses: analyses.length,
      totalDetected,
      averageDetected:
        analyses.length > 0
          ? (totalDetected / analyses.length).toFixed(1)
          : "0.0",
      topConditions: Object.values(conditionCounts)
        .sort((a, b) => b.count - a.count)
        .slice(0, 5),
    };
  }, [analyses]);

  const latestAnalysis = analyses[0];

  return (
    <main className="min-h-[calc(100vh-76px)] bg-[#F7F2EB] px-5 md:px-8 py-10 text-[#2F3728]">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6 mb-8">
          <div>
            <p className="text-sm font-semibold text-[#6F7D55] mb-2">
              Overview
            </p>
            <h1 className="text-3xl md:text-4xl font-bold">
              NetraX Dashboard
            </h1>
            <p className="text-[#69705F] mt-2 max-w-2xl">
              Review your stored retinal image analyses and recent model activity.
            </p>
          </div>

          <button
            onClick={onAnalysis}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-[#8B9A6E] px-5 py-3 text-sm font-semibold text-[#F7F2EB] hover:bg-[#6F7D55] transition"
          >
            <ScanLine size={17} />
            New Analysis
          </button>
        </div>

        {loading && (
          <div className="bg-[#EEEEEE] border border-[#D8D2C8] rounded-2xl p-12 text-center">
            <div className="animate-spin w-10 h-10 border-4 border-[#D8D2C8] border-t-[#8B9A6E] rounded-full mx-auto mb-4" />
            <p className="text-[#69705F]">Loading dashboard...</p>
          </div>
        )}

        {!loading && error && (
          <div className="p-5 rounded-xl bg-[#F2E3DE] border border-[#D5B9B0] text-[#784F43]">
            {error}
          </div>
        )}

        {!loading && !error && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 mb-6">
              <StatCard label="Total Analyses" value={stats.totalAnalyses} icon={FileImage} />
              <StatCard label="Detected Conditions" value={stats.totalDetected} icon={Activity} />
              <StatCard label="Avg. Detected / Scan" value={stats.averageDetected} icon={BarChart3} />
              <StatCard label="Stored Records" value={analyses.length} icon={ShieldCheck} />
            </div>

            <div className="grid lg:grid-cols-5 gap-6">
              <section className="lg:col-span-3 bg-[#EEEEEE] border border-[#D8D2C8] rounded-2xl p-6">
                <div className="flex items-center justify-between gap-4 mb-5">
                  <div>
                    <p className="text-xs uppercase tracking-wider text-[#7C8175]">
                      Latest Activity
                    </p>
                    <h2 className="text-xl font-semibold mt-1">
                      Most recent analysis
                    </h2>
                  </div>
                  <Clock3 size={20} className="text-[#8B9A6E]" />
                </div>

                {latestAnalysis ? (
                  <div className="rounded-xl bg-[#F7F2EB] border border-[#D8D2C8] p-5">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="font-semibold break-all">
                          {latestAnalysis.filename}
                        </p>
                        <div className="flex flex-wrap gap-3 mt-2 text-xs text-[#7C8175]">
                          <span>{latestAnalysis.image_format}</span>
                          <span>
                            {latestAnalysis.image_size?.width} ×{" "}
                            {latestAnalysis.image_size?.height}
                          </span>
                          <span className="flex items-center gap-1">
                            <Calendar size={12} />
                            {new Date(latestAnalysis.created_at).toLocaleString()}
                          </span>
                        </div>
                      </div>

                      <div className="text-right shrink-0">
                        <p className="text-2xl font-bold text-[#6F7D55]">
                          {latestAnalysis.detected_conditions?.length || 0}
                        </p>
                        <p className="text-xs text-[#7C8175]">detected</p>
                      </div>
                    </div>

                    {latestAnalysis.detected_conditions?.length > 0 && (
                      <div className="mt-5 pt-4 border-t border-[#D8D2C8]">
                        <p className="text-xs text-[#7C8175] mb-2">
                          Conditions crossing thresholds
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {latestAnalysis.detected_conditions.slice(0, 6).map((condition) => (
                            <span
                              key={condition.label}
                              className="px-2.5 py-1 rounded-lg bg-[#EAE2D6] border border-[#C8D0B8] text-[#5D694B] text-xs font-semibold"
                            >
                              {condition.label}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    <p className="text-xs text-[#7C8175] mt-4">
                      Model: {latestAnalysis.model}
                    </p>
                  </div>
                ) : (
                  <div className="rounded-xl bg-[#F7F2EB] border border-[#D8D2C8] p-8 text-center">
                    <FileImage size={38} className="mx-auto mb-3 text-[#A5AA9C]" />
                    <p className="text-[#69705F]">
                      No analyses have been recorded yet.
                    </p>
                  </div>
                )}
              </section>

              <section className="lg:col-span-2 bg-[#EEEEEE] border border-[#D8D2C8] rounded-2xl p-6">
                <p className="text-xs uppercase tracking-wider text-[#7C8175]">
                  Across Stored Analyses
                </p>
                <h2 className="text-xl font-semibold mt-1 mb-5">
                  Frequent detections
                </h2>

                {stats.topConditions.length > 0 ? (
                  <div className="space-y-4">
                    {stats.topConditions.map((condition) => {
                      const percentage =
                        stats.totalAnalyses > 0
                          ? (condition.count / stats.totalAnalyses) * 100
                          : 0;

                      return (
                        <div key={condition.label}>
                          <div className="flex items-center justify-between gap-3 mb-2">
                            <div>
                              <p className="text-sm font-medium">{condition.name}</p>
                              <p className="text-xs text-[#7C8175]">
                                {condition.label}
                              </p>
                            </div>
                            <span className="text-sm text-[#69705F]">
                              {condition.count}
                            </span>
                          </div>
                          <div className="h-1.5 rounded-full bg-[#EAE2D6] overflow-hidden">
                            <div
                              className="h-full bg-[#8B9A6E]"
                              style={{ width: `${Math.min(percentage, 100)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="py-10 text-center">
                    <BarChart3 size={36} className="mx-auto mb-3 text-[#A5AA9C]" />
                    <p className="text-sm text-[#69705F]">
                      No detected conditions to summarize yet.
                    </p>
                  </div>
                )}

                <button
                  onClick={onHistory}
                  className="w-full mt-6 inline-flex items-center justify-center gap-2 rounded-xl border border-[#D1CBC1] bg-[#F7F2EB] px-4 py-2.5 text-sm font-semibold text-[#4D5545] hover:border-[#8B9A6E] hover:bg-[#EAE2D6] transition"
                >
                  <History size={16} />
                  View Full History
                  <ArrowRight size={15} />
                </button>
              </section>
            </div>

            <div className="mt-6 p-4 rounded-xl bg-[#EAE2D6] border border-[#D8D2C8]">
              <p className="text-xs leading-relaxed text-[#69705F]">
                Dashboard statistics are calculated from your stored NetraX model
                outputs. They describe model detections and are not clinical diagnoses.
              </p>
            </div>
          </>
        )}
      </div>
    </main>
  );
}

function StatCard({ label, value, icon: Icon }) {
  return (
    <div className="bg-[#EEEEEE] border border-[#D8D2C8] rounded-2xl p-5">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-wider text-[#7C8175]">
            {label}
          </p>
          <p className="text-3xl font-bold mt-2">{value}</p>
        </div>
        <div className="p-3 rounded-xl bg-[#EAE2D6] text-[#6F7D55]">
          <Icon size={22} />
        </div>
      </div>
    </div>
  );
}
