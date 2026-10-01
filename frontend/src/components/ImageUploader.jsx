import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  FileImage,
  Info,
  RefreshCw,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";
import { predictImage } from "../services/api";

const MAX_FILE_SIZE_MB = 10;
const ACCEPTED_TYPES = [
  "image/png",
  "image/jpeg",
  "image/jpg",
  "image/webp",
];

export default function ImageUploader() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  const detectedConditions = result?.results?.detected_conditions || [];
  const allPredictions = result?.results?.all_predictions || [];

  const notDetectedConditions = useMemo(
    () => allPredictions.filter((condition) => !condition.detected),
    [allPredictions]
  );

  const highestDetectedProbability =
    detectedConditions.length > 0
      ? Math.max(
          ...detectedConditions.map((condition) => condition.probability)
        )
      : 0;

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    setError("");

    if (!ACCEPTED_TYPES.includes(selectedFile.type)) {
      setError("Unsupported file type. Please upload PNG, JPG, JPEG, or WebP.");
      return;
    }

    const sizeMb = selectedFile.size / (1024 * 1024);

    if (sizeMb > MAX_FILE_SIZE_MB) {
      setError(`File is too large. Maximum supported size is ${MAX_FILE_SIZE_MB} MB.`);
      return;
    }

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
  };

  const handleFileChange = (event) => {
    handleFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    handleFile(event.dataTransfer.files?.[0]);
  };

  const handlePredict = async () => {
    if (!file) {
      setError("Please select a retinal image first.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setResult(null);

      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      console.error("NETRAX API ERROR:", err);

      if (err.response) {
        if (err.response.status === 401) {
          setError("Your session has expired. Please sign in again.");
        } else {
          setError(
            `API Error ${err.response.status}: ${
              err.response.data?.detail || "Prediction failed."
            }`
          );
        }
      } else if (err.request) {
        setError("No response from the NetraX backend.");
      } else {
        setError(`Request Error: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const clearAnalysis = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
  };

  return (
    <main className="min-h-[calc(100vh-76px)] bg-[#F7F2EB] text-[#2F3728] px-5 md:px-8 py-8 md:py-10">
      <div className="max-w-7xl mx-auto">
        {/* Page heading */}
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-5 mb-8">
          <div>
            <p className="text-sm font-semibold text-[#6F7D55] mb-2">
              Retinal analysis
            </p>
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
              Analyze a retinal image
            </h1>
            <p className="text-[#69705F] mt-2 max-w-2xl">
              Upload a retinal fundus image and review the trained NetraX
              model&apos;s threshold-aware outputs.
            </p>
          </div>

          {result && (
            <button
              onClick={clearAnalysis}
              className="inline-flex items-center justify-center gap-2 rounded-xl border border-[#C8C2B8] bg-[#EEEEEE] px-4 py-2.5 text-sm font-semibold text-[#4F5847] hover:border-[#8B9A6E] hover:text-[#5F6D49] transition"
            >
              <RefreshCw size={16} />
              New analysis
            </button>
          )}
        </div>

        {/* Main workspace */}
        <div className="grid xl:grid-cols-[0.9fr_1.1fr] gap-6">
          {/* Upload panel */}
          <section className="bg-[#EEEEEE] border border-[#D8D2C8] rounded-3xl p-5 md:p-6 shadow-sm">
            <div className="flex items-start justify-between gap-4 mb-5">
              <div>
                <div className="flex items-center gap-2">
                  <div className="p-2 rounded-lg bg-[#8B9A6E] text-[#F7F2EB]">
                    <FileImage size={18} />
                  </div>
                  <h2 className="text-xl font-bold">Retinal image</h2>
                </div>
                <p className="text-sm text-[#69705F] mt-2">
                  Supported formats: PNG, JPG, JPEG, WebP · Maximum 10 MB
                </p>
              </div>

              {file && (
                <button
                  onClick={clearAnalysis}
                  title="Remove image"
                  className="p-2 rounded-lg text-[#7C8175] hover:bg-[#EAE2D6] hover:text-[#4F5847] transition"
                >
                  <X size={18} />
                </button>
              )}
            </div>

            <label
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`block cursor-pointer rounded-2xl border-2 border-dashed p-7 md:p-8 text-center transition ${
                isDragging
                  ? "border-[#8B9A6E] bg-[#EAE2D6]"
                  : "border-[#C8C2B8] bg-[#F7F2EB] hover:border-[#8B9A6E] hover:bg-[#F3EEE6]"
              }`}
            >
              <div className="w-14 h-14 mx-auto rounded-2xl bg-[#EAE2D6] flex items-center justify-center mb-4">
                <Upload size={28} className="text-[#6F7D55]" />
              </div>

              <p className="font-semibold">
                {isDragging ? "Drop image here" : "Choose a retinal image"}
              </p>

              <p className="text-sm text-[#7C8175] mt-1">
                Drag and drop or click to browse
              </p>

              {file && (
                <div className="mt-4 inline-flex max-w-full items-center gap-2 rounded-lg bg-[#EAE2D6] px-3 py-2">
                  <FileImage size={15} className="shrink-0 text-[#6F7D55]" />
                  <span className="text-sm font-medium text-[#4F5847] truncate">
                    {file.name}
                  </span>
                </div>
              )}

              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg,image/webp"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>

            {/* Preview */}
            <div className="mt-5">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-semibold text-[#4F5847]">
                  Image preview
                </p>
                {file && (
                  <p className="text-xs text-[#7C8175]">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                )}
              </div>

              <div className="min-h-64 md:min-h-80 rounded-2xl border border-[#D8D2C8] bg-[#F7F2EB] overflow-hidden flex items-center justify-center">
                {preview ? (
                  <img
                    src={preview}
                    alt="Selected retinal fundus preview"
                    className="w-full h-full max-h-[420px] object-contain"
                  />
                ) : (
                  <div className="text-center px-6 py-10">
                    <Brain size={42} className="mx-auto mb-3 text-[#B0B5A9]" />
                    <p className="text-sm font-medium text-[#69705F]">
                      Your retinal image will appear here
                    </p>
                  </div>
                )}
              </div>
            </div>

            <button
              onClick={handlePredict}
              disabled={!file || loading}
              className="w-full mt-5 py-3.5 rounded-xl bg-[#8B9A6E] hover:bg-[#6F7D55] disabled:bg-[#D1CBC1] disabled:text-[#969A90] text-[#F7F2EB] font-semibold transition shadow-sm"
            >
              {loading ? (
                <span className="inline-flex items-center justify-center gap-2">
                  <span className="animate-spin h-4 w-4 rounded-full border-2 border-[#EAE2D6] border-t-transparent" />
                  Analyzing retinal image...
                </span>
              ) : (
                "Analyze Image"
              )}
            </button>

            {error && (
              <div className="mt-4 p-4 rounded-xl bg-[#F2E3DE] border border-[#D5B9B0] text-[#784F43]">
                <div className="flex items-start gap-3">
                  <AlertTriangle size={18} className="shrink-0 mt-0.5" />
                  <p className="text-sm leading-relaxed">{error}</p>
                </div>
              </div>
            )}

            <div className="mt-5 p-4 rounded-xl bg-[#EAE2D6] border border-[#D8D2C8]">
              <div className="flex items-start gap-3">
                <ShieldCheck size={18} className="shrink-0 text-[#6F7D55] mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-[#4F5847]">
                    Decision-support workflow
                  </p>
                  <p className="text-xs leading-relaxed text-[#69705F] mt-1">
                    NetraX presents model outputs for research and decision
                    support. It does not establish a clinical diagnosis.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Results panel */}
          <section className="bg-[#EEEEEE] border border-[#D8D2C8] rounded-3xl p-5 md:p-6 shadow-sm">
            <div className="flex items-start justify-between gap-4 mb-5">
              <div>
                <div className="flex items-center gap-2">
                  <div className="p-2 rounded-lg bg-[#EAE2D6] text-[#6F7D55]">
                    <BarChart3 size={18} />
                  </div>
                  <h2 className="text-xl font-bold">Analysis results</h2>
                </div>
                <p className="text-sm text-[#69705F] mt-2">
                  Threshold-aware model outputs from the current analysis.
                </p>
              </div>

              {result && (
                <span className="hidden sm:inline-flex items-center gap-1.5 rounded-full bg-[#EAE2D6] px-3 py-1.5 text-xs font-semibold text-[#5F6D49]">
                  <CheckCircle2 size={14} />
                  Analysis complete
                </span>
              )}
            </div>

            {!result && !loading && (
              <EmptyResults />
            )}

            {loading && <LoadingResults />}

            {result && !loading && (
              <div>
                {/* Metadata */}
                <div className="grid grid-cols-2 xl:grid-cols-3 gap-3 mb-5">
                  <ResultMeta label="Model" value={result.model} />
                  <ResultMeta
                    label="Detected"
                    value={detectedConditions.length}
                    accent
                    note="thresholds crossed"
                  />
                  <ResultMeta
                    label="Image"
                    value={`${result.image_size?.width} × ${result.image_size?.height}`}
                    note={`${result.image_format} · ${result.filename}`}
                  />
                </div>

                {/* Summary */}
                <div className="rounded-2xl border border-[#C8D0B8] bg-[#EAE2D6] p-5 mb-5">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <div>
                      <p className="text-xs uppercase tracking-wider text-[#7C8175]">
                        Highest detected probability
                      </p>
                      <p className="text-3xl font-bold text-[#5F6D49] mt-1">
                        {(highestDetectedProbability * 100).toFixed(2)}%
                      </p>
                    </div>

                    <div className="w-fit p-3 rounded-xl bg-[#F7F2EB] text-[#8B9A6E]">
                      <Brain size={26} />
                    </div>
                  </div>

                  <p className="text-xs leading-relaxed text-[#69705F] mt-3">
                    Model probability is not clinical certainty. A detection
                    means the output crossed the configured model threshold.
                  </p>
                </div>

                {/* Detected conditions */}
                {detectedConditions.length > 0 ? (
                  <section>
                    <div className="flex items-center justify-between gap-3 mb-3">
                      <div className="flex items-center gap-2">
                        <AlertTriangle size={18} className="text-[#8B9A6E]" />
                        <h3 className="font-bold">
                          Conditions crossing thresholds
                        </h3>
                      </div>

                      <span className="text-xs font-semibold text-[#7C8175]">
                        {detectedConditions.length} detected
                      </span>
                    </div>

                    <div className="space-y-3">
                      {detectedConditions.map((condition) => (
                        <ConditionCard
                          key={condition.label}
                          condition={condition}
                          detected
                        />
                      ))}
                    </div>
                  </section>
                ) : (
                  <div className="rounded-2xl border border-[#C8D0B8] bg-[#EAE2D6] p-5">
                    <div className="flex items-start gap-3">
                      <CheckCircle2
                        className="text-[#6F7D55] shrink-0"
                        size={23}
                      />
                      <div>
                        <p className="font-bold">No thresholds crossed</p>
                        <p className="text-sm text-[#69705F] mt-1 leading-relaxed">
                          No modeled condition crossed its configured detection
                          threshold for this image.
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Other predictions */}
                {notDetectedConditions.length > 0 && (
                  <details className="mt-5 group">
                    <summary className="cursor-pointer list-none flex items-center justify-between gap-4 rounded-xl border border-[#D8D2C8] bg-[#EAE2D6] p-4 hover:border-[#B8C0A7] transition">
                      <div>
                        <p className="font-bold">
                          Other model predictions
                        </p>
                        <p className="text-xs text-[#7C8175] mt-1">
                          {notDetectedConditions.length} outputs below their
                          configured thresholds
                        </p>
                      </div>

                      <span className="text-[#69705F] group-open:rotate-180 transition-transform">
                        ▼
                      </span>
                    </summary>

                    <div className="mt-3 space-y-2">
                      {notDetectedConditions.map((condition) => (
                        <ConditionCard
                          key={condition.label}
                          condition={condition}
                        />
                      ))}
                    </div>
                  </details>
                )}

                {/* Model information */}
                <div className="mt-5 rounded-2xl border border-[#D8D2C8] bg-[#F7F2EB] p-4">
                  <div className="flex items-start gap-3">
                    <Info size={18} className="text-[#8B9A6E] shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-bold">
                        How to read these results
                      </p>
                      <p className="text-xs leading-relaxed text-[#69705F] mt-1">
                        Each output has its own configured threshold. The
                        probability indicates the model&apos;s output for that
                        label; it should not be interpreted as a probability
                        of clinical disease.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="mt-4 p-4 rounded-xl bg-[#EAE2D6] border border-[#D8D2C8]">
                  <p className="text-xs leading-relaxed text-[#69705F]">
                    NetraX provides AI-assisted analysis for research and
                    decision support. Results are not a medical diagnosis and
                    should not replace evaluation by a qualified healthcare
                    professional.
                  </p>
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}

function ResultMeta({ label, value, note, accent = false }) {
  return (
    <div className="p-4 rounded-xl bg-[#F7F2EB] border border-[#D8D2C8] min-w-0">
      <p className="text-xs uppercase tracking-wider text-[#7C8175]">
        {label}
      </p>

      <p
        className={`font-semibold mt-1 break-words ${
          accent ? "text-2xl text-[#5F6D49]" : "text-sm text-[#2F3728]"
        }`}
      >
        {value}
      </p>

      {note && (
        <p className="text-xs text-[#7C8175] mt-1 break-words">
          {note}
        </p>
      )}
    </div>
  );
}

function ConditionCard({ condition, detected = false }) {
  const probability = Number(condition.probability || 0);
  const confidence = Number(condition.confidence_percent || 0);
  const threshold = Number(condition.threshold || 0);

  const thresholdPercent = Math.min(Math.max(threshold * 100, 0), 100);
  const probabilityPercent = Math.min(Math.max(confidence, 0), 100);

  return (
    <div
      className={`rounded-2xl p-4 border ${
        detected
          ? "bg-[#F7F2EB] border-[#C8D0B8]"
          : "bg-[#F7F2EB] border-[#D8D2C8]"
      }`}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <p className="font-bold">{condition.name}</p>
            <span className="text-xs px-2 py-1 rounded-md bg-[#EAE2D6] text-[#69705F]">
              {condition.label}
            </span>
          </div>

          {condition.description && (
            <p className="text-sm text-[#69705F] mt-1 leading-relaxed">
              {condition.description}
            </p>
          )}
        </div>

        <div className="text-right shrink-0">
          <p
            className={`font-bold ${
              detected ? "text-[#5F6D49]" : "text-[#69705F]"
            }`}
          >
            {confidence.toFixed(2)}%
          </p>
          <p className="text-[10px] text-[#7C8175]">
            model output
          </p>
        </div>
      </div>

      <div className="mt-4">
        <div className="relative h-2.5 rounded-full bg-[#EAE2D6] overflow-visible">
          <div
            className={`h-full rounded-full ${
              detected ? "bg-[#8B9A6E]" : "bg-[#A5AE91]"
            }`}
            style={{ width: `${probabilityPercent}%` }}
          />

          <div
            className="absolute top-1/2 -translate-y-1/2 w-0.5 h-4 bg-[#4F5847]"
            style={{ left: `${thresholdPercent}%` }}
            title={`Threshold ${threshold.toFixed(2)}`}
          />
        </div>

        <div className="flex items-center justify-between mt-2 text-[11px] text-[#7C8175]">
          <span>
            Threshold: {threshold.toFixed(2)}
          </span>
          <span>
            {detected ? "Threshold crossed" : "Below threshold"}
          </span>
        </div>
      </div>
    </div>
  );
}

function EmptyResults() {
  return (
    <div className="min-h-[620px] rounded-2xl border border-[#D8D2C8] bg-[#F7F2EB] flex items-center justify-center text-center px-8">
      <div className="max-w-sm">
        <div className="w-16 h-16 mx-auto rounded-2xl bg-[#EAE2D6] flex items-center justify-center mb-5">
          <Brain size={34} className="text-[#8B9A6E]" />
        </div>

        <h3 className="text-xl font-bold">
          Ready for analysis
        </h3>

        <p className="text-sm text-[#69705F] leading-relaxed mt-2">
          Upload a retinal fundus image on the left to run the NetraX
          inference pipeline and review its model outputs here.
        </p>

        <div className="mt-5 flex items-center justify-center gap-2 text-xs text-[#7C8175]">
          <ShieldCheck size={14} className="text-[#8B9A6E]" />
          Decision-support workflow
        </div>
      </div>
    </div>
  );
}

function LoadingResults() {
  return (
    <div className="min-h-[620px] rounded-2xl border border-[#D8D2C8] bg-[#F7F2EB] flex items-center justify-center text-center">
      <div>
        <div className="w-12 h-12 mx-auto rounded-full border-4 border-[#D8D2C8] border-t-[#8B9A6E] animate-spin mb-5" />
        <h3 className="font-bold">Running NetraX inference</h3>
        <p className="text-sm text-[#69705F] mt-2">
          Processing the retinal image and generating model outputs...
        </p>
      </div>
    </div>
  );
}


