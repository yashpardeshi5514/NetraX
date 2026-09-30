import {
  ArrowRight,
  Brain,
  Database,
  History,
  LockKeyhole,
  ScanLine,
  ShieldCheck,
  Sparkles,
  Upload,
} from "lucide-react";

export default function LandingPage({ onGetStarted, onSignIn }) {
  return (
    <div className="min-h-screen bg-[#F7F2EB] text-[#2F3728] overflow-hidden">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 left-1/2 h-96 w-96 -translate-x-1/2 rounded-full bg-[#8B9A6E]/10 blur-3xl" />
        <div className="absolute top-72 -left-40 h-80 w-80 rounded-full bg-[#EAE2D6] blur-3xl" />
        <div className="absolute top-[32rem] -right-40 h-80 w-80 rounded-full bg-[#8B9A6E]/10 blur-3xl" />
      </div>

      <header className="relative z-10 border-b border-[#D8D2C8] bg-[#F7F2EB]/90">
        <div className="max-w-7xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            <button onClick={onGetStarted} className="flex items-center gap-3">
              <div className="h-11 w-11 rounded-xl bg-[#8B9A6E] text-[#F7F2EB] flex items-center justify-center shadow-sm">
                <Brain size={23} />
              </div>
              <div className="text-left">
                <p className="font-bold text-lg leading-tight">NetraX</p>
                <p className="text-xs text-[#69705F]">
                  Retinal AI Decision Support
                </p>
              </div>
            </button>

            <div className="flex items-center gap-2 md:gap-3">
              <button
                onClick={onSignIn}
                className="px-4 py-2 rounded-xl text-sm font-semibold text-[#4D5545] hover:bg-[#EAE2D6] transition"
              >
                Sign In
              </button>
              <button
                onClick={onGetStarted}
                className="inline-flex items-center gap-2 rounded-xl bg-[#8B9A6E] px-4 py-2.5 text-sm font-semibold text-[#F7F2EB] hover:bg-[#6F7D55] transition"
              >
                Get Started
                <ArrowRight size={15} />
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="relative z-10">
        <section className="max-w-7xl mx-auto px-6 pt-20 pb-24 md:pt-28 md:pb-32">
          <div className="max-w-4xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-[#C8D0B8] bg-[#EAE2D6] text-[#5D694B] text-xs font-semibold mb-7">
              <Sparkles size={14} />
              AI-assisted retinal image analysis
            </div>

            <h1 className="text-5xl md:text-7xl font-bold tracking-tight leading-[1.05]">
              Turn retinal images into
              <span className="block text-[#6F7D55] mt-2">
                structured insights.
              </span>
            </h1>

            <p className="max-w-2xl mx-auto mt-7 text-lg md:text-xl leading-relaxed text-[#69705F]">
              NetraX analyzes retinal fundus images with a trained
              multi-label deep learning model and presents model outputs
              as clear, interpretable decision-support information.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-3 mt-9">
              <button
                onClick={onGetStarted}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl bg-[#8B9A6E] px-6 py-3.5 font-semibold text-[#F7F2EB] hover:bg-[#6F7D55] transition shadow-sm"
              >
                Start Analysis
                <ArrowRight size={18} />
              </button>
              <button
                onClick={onSignIn}
                className="w-full sm:w-auto rounded-xl border border-[#CFC8BD] bg-[#EEEEEE] px-6 py-3.5 font-semibold text-[#3E4637] hover:bg-[#EAE2D6] hover:border-[#8B9A6E] transition"
              >
                Sign In
              </button>
            </div>

            <p className="mt-5 text-xs text-[#7C8175]">
              Research and decision-support tool · Not a medical diagnosis
            </p>
          </div>

          <div className="max-w-5xl mx-auto mt-16">
            <div className="rounded-3xl border border-[#D8D2C8] bg-[#EAE2D6] p-2 shadow-xl shadow-[#8B9A6E]/10">
              <div className="rounded-2xl border border-[#D8D2C8] bg-[#F7F2EB] overflow-hidden">
                <div className="flex items-center gap-2 border-b border-[#D8D2C8] px-5 py-3">
                  <span className="h-2.5 w-2.5 rounded-full bg-[#C8C2B8]" />
                  <span className="h-2.5 w-2.5 rounded-full bg-[#C8C2B8]" />
                  <span className="h-2.5 w-2.5 rounded-full bg-[#C8C2B8]" />
                  <div className="ml-4 h-6 flex-1 max-w-md rounded-md bg-[#EEEEEE] border border-[#D8D2C8]" />
                </div>

                <div className="grid md:grid-cols-3 gap-4 p-5 md:p-7">
                  <div className="md:col-span-2 rounded-2xl border border-[#D8D2C8] bg-[#EEEEEE] p-5">
                    <div className="flex items-center gap-3 mb-5">
                      <div className="p-2 rounded-lg bg-[#8B9A6E] text-[#F7F2EB]">
                        <ScanLine size={18} />
                      </div>
                      <div>
                        <p className="font-semibold">Retinal Analysis</p>
                        <p className="text-xs text-[#69705F]">
                          Model-generated decision support
                        </p>
                      </div>
                    </div>

                    <div className="h-44 md:h-56 rounded-xl border border-[#D8D2C8] bg-[#F7F2EB] flex items-center justify-center">
                      <div className="text-center">
                        <Upload size={34} className="mx-auto mb-3 text-[#8B9A6E]" />
                        <p className="text-sm font-medium text-[#3E4637]">
                          Upload a retinal fundus image
                        </p>
                        <p className="text-xs text-[#7C8175] mt-1">
                          PNG · JPG · WebP
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-[#D8D2C8] bg-[#EEEEEE] p-5">
                    <p className="text-xs uppercase tracking-wider text-[#7C8175]">
                      Example output
                    </p>

                    <div className="mt-5 space-y-3">
                      <div className="rounded-xl border border-[#C8D0B8] bg-[#EAE2D6] p-4">
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-sm font-medium">
                            Model output
                          </span>
                          <span className="text-[#6F7D55] text-sm font-semibold">
                            97.2%
                          </span>
                        </div>
                        <p className="text-xs text-[#69705F] mt-1">
                          Probability shown for illustration
                        </p>
                      </div>

                      <div className="rounded-xl border border-[#D8D2C8] bg-[#F7F2EB] p-4">
                        <p className="text-sm font-medium">
                          Threshold-aware results
                        </p>
                        <p className="text-xs text-[#69705F] mt-1">
                          Each condition uses its configured detection threshold.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="border-y border-[#D8D2C8] bg-[#EAE2D6]/60">
          <div className="max-w-7xl mx-auto px-6 py-20">
            <div className="max-w-2xl mb-12">
              <p className="text-sm font-semibold text-[#6F7D55] mb-2">
                Built for structured analysis
              </p>
              <h2 className="text-3xl md:text-4xl font-bold">
                One workflow from image to stored analysis.
              </h2>
              <p className="text-[#69705F] mt-4 leading-relaxed">
                NetraX combines authenticated access, image inference,
                threshold-aware results, and persistent analysis history
                in one application.
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              <FeatureCard icon={Brain} title="AI Inference" text="Run retinal images through the trained NetraX multi-label model." />
              <FeatureCard icon={ScanLine} title="45 Outputs" text="Evaluate the model's 45 retinal disease and finding labels." />
              <FeatureCard icon={Database} title="Stored History" text="Keep authenticated analysis results in your PostgreSQL-backed history." />
              <FeatureCard icon={ShieldCheck} title="Decision Support" text="Present model probabilities and thresholds without framing them as diagnosis." />
            </div>
          </div>
        </section>

        <section className="max-w-7xl mx-auto px-6 py-20">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <p className="text-sm font-semibold text-[#6F7D55] mb-2">
                Simple workflow
              </p>
              <h2 className="text-3xl md:text-4xl font-bold">
                From upload to analysis history.
              </h2>
              <p className="text-[#69705F] mt-4 leading-relaxed">
                Sign in, upload a retinal image, review the model output,
                and return to your stored analyses whenever needed.
              </p>
            </div>

            <div className="space-y-3">
              <WorkflowStep number="01" title="Authenticate" text="Sign in to your protected NetraX workspace." />
              <WorkflowStep number="02" title="Upload" text="Provide a supported retinal fundus image." />
              <WorkflowStep number="03" title="Analyze" text="Run the image through the trained inference pipeline." />
              <WorkflowStep number="04" title="Review" text="Inspect threshold-aware results and stored history." />
            </div>
          </div>
        </section>

        <section className="max-w-7xl mx-auto px-6 pb-20">
          <div className="rounded-3xl border border-[#C8D0B8] bg-[#EAE2D6] p-8 md:p-12 text-center">
            <div className="mx-auto w-fit p-3 rounded-xl bg-[#8B9A6E] text-[#F7F2EB] mb-5">
              <Brain size={25} />
            </div>
            <h2 className="text-3xl font-bold">
              Ready to analyze a retinal image?
            </h2>
            <p className="max-w-xl mx-auto text-[#69705F] mt-3">
              Start a NetraX session and explore AI-assisted retinal image analysis.
            </p>
            <button
              onClick={onGetStarted}
              className="mt-7 inline-flex items-center gap-2 rounded-xl bg-[#8B9A6E] px-6 py-3.5 font-semibold text-[#F7F2EB] hover:bg-[#6F7D55] transition"
            >
              Get Started
              <ArrowRight size={18} />
            </button>
          </div>
        </section>
      </main>

      <footer className="border-t border-[#D8D2C8]">
        <div className="max-w-7xl mx-auto px-6 py-7 flex flex-col md:flex-row items-center justify-between gap-3">
          <p className="text-sm text-[#7C8175]">© {new Date().getFullYear()} NetraX</p>
          <div className="flex items-center gap-2 text-xs text-[#7C8175]">
            <LockKeyhole size={13} />
            <span>Authenticated decision-support platform</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon: Icon, title, text }) {
  return (
    <div className="rounded-2xl border border-[#D8D2C8] bg-[#F7F2EB] p-5 hover:border-[#8B9A6E] transition">
      <div className="w-fit p-2.5 rounded-xl bg-[#EAE2D6] text-[#6F7D55] mb-4">
        <Icon size={20} />
      </div>
      <h3 className="font-semibold">{title}</h3>
      <p className="text-sm leading-relaxed text-[#69705F] mt-2">{text}</p>
    </div>
  );
}

function WorkflowStep({ number, title, text }) {
  return (
    <div className="flex gap-4 rounded-2xl border border-[#D8D2C8] bg-[#F7F2EB] p-5">
      <div className="shrink-0 flex items-center justify-center w-10 h-10 rounded-xl bg-[#EAE2D6] border border-[#C8D0B8] text-[#6F7D55] text-sm font-bold">
        {number}
      </div>
      <div>
        <h3 className="font-semibold">{title}</h3>
        <p className="text-sm text-[#69705F] mt-1">{text}</p>
      </div>
    </div>
  );
}
