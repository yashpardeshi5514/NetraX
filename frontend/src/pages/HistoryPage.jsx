import { useEffect, useState } from "react";
import {
  Brain,
  Calendar,
  ChevronDown,
  Image as ImageIcon,
} from "lucide-react";
import { getAnalysisHistory } from "../services/api";

export default function HistoryPage() {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        setLoading(true);
        setError("");
        const data = await getAnalysisHistory();
        setAnalyses(data.analyses || []);
      } catch (err) {
        console.error("HISTORY API ERROR:", err);
        setError(
          err.response?.data?.detail ||
            "Unable to load analysis history."
        );
      } finally {
        setLoading(false);
      }
    };

    loadHistory();
  }, []);

  return (
    <main className="min-h-[calc(100vh-76px)] bg-[#F7F2EB] text-[#2F3728] px-5 md:px-8 py-10">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-3 rounded-xl bg-[#8B9A6E] text-[#F7F2EB]">
            <Brain size={26} />
          </div>
          <div>
            <h1 className="text-3xl font-bold">Analysis History</h1>
            <p className="text-[#69705F] text-sm">
              Previous retinal image analyses
            </p>
          </div>
        </div>

        {loading && (
          <div className="bg-[#EEEEEE] border border-[#D8D2C8] rounded-2xl p-10 text-center">
            <div className="animate-spin w-10 h-10 border-4 border-[#D8D2C8] border-t-[#8B9A6E] rounded-full mx-auto mb-4" />
            <p className="text-[#69705F]">Loading analysis history...</p>
          </div>
        )}

        {!loading && error && (
          <div className="p-5 rounded-xl bg-[#F2E3DE] border border-[#D5B9B0] text-[#784F43]">
            {error}
          </div>
        )}

        {!loading && !error && analyses.length === 0 && (
          <div className="bg-[#EEEEEE] border border-[#D8D2C8] rounded-2xl p-12 text-center">
            <ImageIcon size={48} className="mx-auto mb-4 text-[#A5AA9C]" />
            <h2 className="text-xl font-semibold mb-2">No analyses yet</h2>
            <p className="text-[#69705F]">
              Upload a retinal image to create your first analysis.
            </p>
          </div>
        )}

        {!loading && !error && analyses.length > 0 && (
          <div className="space-y-4">
            {analyses.map((analysis) => {
              const isExpanded = expandedId === analysis.id;

              return (
                <div
                  key={analysis.id}
                  className="bg-[#EEEEEE] border border-[#D8D2C8] rounded-2xl overflow-hidden"
                >
                  <button
                    onClick={() =>
                      setExpandedId(isExpanded ? null : analysis.id)
                    }
                    className="w-full text-left p-5 hover:bg-[#EAE2D6]/60 transition"
                  >
                    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                      <div>
                        <div className="flex items-center gap-3 mb-2">
                          <ImageIcon size={20} className="text-[#6F7D55]" />
                          <h2 className="font-semibold">{analysis.filename}</h2>
                        </div>

                        <div className="flex flex-wrap gap-4 text-xs text-[#7C8175]">
                          <span>{analysis.image_format}</span>
                          <span>
                            {analysis.image_size?.width} ×{" "}
                            {analysis.image_size?.height}
                          </span>
                          <span className="flex items-center gap-1">
                            <Calendar size={13} />
                            {new Date(analysis.created_at).toLocaleString()}
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center gap-6">
                        <div className="text-right">
                          <p className="text-xs uppercase tracking-wider text-[#7C8175]">
                            Detected
                          </p>
                          <p className="text-2xl font-bold text-[#6F7D55]">
                            {analysis.detected_conditions?.length || 0}
                          </p>
                          <p className="text-xs text-[#7C8175]">conditions</p>
                        </div>

                        <ChevronDown
                          size={20}
                          className={`text-[#7C8175] transition-transform ${
                            isExpanded ? "rotate-180" : ""
                          }`}
                        />
                      </div>
                    </div>
                  </button>

                  {isExpanded && (
                    <div className="border-t border-[#D8D2C8] p-5">
                      {analysis.detected_conditions?.length > 0 ? (
                        <div className="mb-6">
                          <p className="text-xs uppercase tracking-wider text-[#7C8175] mb-3">
                            Conditions crossing thresholds
                          </p>

                          <div className="space-y-3">
                            {analysis.detected_conditions.map((condition) => (
                              <div
                                key={condition.label}
                                className="p-4 rounded-xl bg-[#EAE2D6] border border-[#C8D0B8]"
                              >
                                <div className="flex items-start justify-between gap-4">
                                  <div>
                                    <p className="font-semibold">{condition.name}</p>
                                    <p className="text-xs text-[#7C8175] mt-1">
                                      {condition.label}
                                    </p>
                                  </div>
                                  <div className="text-right">
                                    <p className="text-[#6F7D55] font-semibold">
                                      {condition.confidence_percent.toFixed(2)}%
                                    </p>
                                    <p className="text-[10px] text-[#7C8175]">
                                      probability
                                    </p>
                                  </div>
                                </div>

                                <div className="w-full h-2 bg-[#F7F2EB] rounded-full overflow-hidden mt-3">
                                  <div
                                    className="h-full bg-[#8B9A6E]"
                                    style={{
                                      width: `${Math.min(
                                        condition.confidence_percent,
                                        100
                                      )}%`,
                                    }}
                                  />
                                </div>

                                <div className="flex justify-between mt-2 text-xs text-[#7C8175]">
                                  <span>
                                    Threshold: {condition.threshold.toFixed(2)}
                                  </span>
                                  <span>Detected</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : (
                        <div className="mb-6 p-4 rounded-xl bg-[#F7F2EB] border border-[#D8D2C8]">
                          <p className="text-sm text-[#69705F]">
                            No modeled condition crossed its configured detection threshold.
                          </p>
                        </div>
                      )}

                      {analysis.all_predictions?.length > 0 && (
                        <details>
                          <summary className="cursor-pointer text-sm font-semibold text-[#4D5545] hover:text-[#2F3728]">
                            View all model predictions
                          </summary>

                          <div className="mt-4 space-y-2">
                            {analysis.all_predictions.map((condition) => (
                              <div
                                key={condition.label}
                                className="p-3 rounded-lg bg-[#F7F2EB] border border-[#D8D2C8]"
                              >
                                <div className="flex items-center justify-between gap-4">
                                  <div>
                                    <span className="font-medium text-sm">
                                      {condition.name}
                                    </span>
                                    <span className="ml-2 text-xs px-2 py-1 rounded-md bg-[#EAE2D6] text-[#69705F]">
                                      {condition.label}
                                    </span>
                                  </div>

                                  <span className="text-sm text-[#69705F]">
                                    {condition.confidence_percent.toFixed(2)}%
                                  </span>
                                </div>

                                <div className="w-full h-1.5 bg-[#EAE2D6] rounded-full overflow-hidden mt-2">
                                  <div
                                    className="h-full bg-[#8B9A6E]"
                                    style={{
                                      width: `${Math.min(
                                        condition.confidence_percent,
                                        100
                                      )}%`,
                                    }}
                                  />
                                </div>

                                <div className="flex justify-between mt-2 text-[11px] text-[#7C8175]">
                                  <span>
                                    Threshold: {condition.threshold.toFixed(2)}
                                  </span>
                                  <span>
                                    {condition.detected
                                      ? "Detected"
                                      : "Below threshold"}
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </details>
                      )}

                      <div className="mt-6 pt-4 border-t border-[#D8D2C8] text-xs text-[#7C8175]">
                        Model: {analysis.model}
                      </div>

                      <div className="mt-4 p-4 rounded-xl bg-[#EAE2D6] border border-[#D8D2C8]">
                        <p className="text-xs leading-relaxed text-[#69705F]">
                          NetraX provides AI-assisted analysis for research and
                          decision support. Results are not a medical diagnosis
                          and should not replace evaluation by a qualified
                          healthcare professional.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}
