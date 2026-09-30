import { useState } from "react";
import { ArrowLeft, Brain, Eye, EyeOff, LogIn, UserPlus } from "lucide-react";
import { loginUser, registerUser } from "../services/auth";

export default function AuthPage({ onAuthenticated, onBackToLanding }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const isLogin = mode === "login";

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setSuccess("");

    if (!email.trim() || !password) {
      setError("Please enter your email and password.");
      return;
    }

    if (!isLogin && password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }

    try {
      setLoading(true);

      if (isLogin) {
        const data = await loginUser(email.trim(), password);
        onAuthenticated(data.user);
      } else {
        await registerUser(email.trim(), password);
        setSuccess("Account created. You can now sign in.");
        setMode("login");
        setPassword("");
      }
    } catch (err) {
      const detail =
        err.response?.data?.detail ||
        "Something went wrong. Please try again.";
      setError(detail);
    } finally {
      setLoading(false);
    }
  };

  const switchMode = (nextMode) => {
    setMode(nextMode);
    setError("");
    setSuccess("");
    setPassword("");
  };

  return (
    <div className="min-h-screen bg-[#F7F2EB] text-[#2F3728] flex items-center justify-center px-6 py-10">
      <div className="w-full max-w-md">
        {onBackToLanding && (
          <button
            onClick={onBackToLanding}
            className="inline-flex items-center gap-2 mb-6 text-sm font-semibold text-[#69705F] hover:text-[#2F3728] transition"
          >
            <ArrowLeft size={16} />
            Back to landing
          </button>
        )}

        <div className="rounded-3xl border border-[#D8D2C8] bg-[#EEEEEE] p-7 md:p-8 shadow-xl shadow-[#8B9A6E]/10">
          <div className="text-center mb-7">
            <div className="mx-auto w-14 h-14 rounded-2xl bg-[#8B9A6E] text-[#F7F2EB] flex items-center justify-center mb-4">
              <Brain size={28} />
            </div>

            <h1 className="text-2xl font-bold">Welcome to NetraX</h1>
            <p className="text-sm text-[#69705F] mt-2">
              {isLogin
                ? "Sign in to access your retinal analysis workspace."
                : "Create an account to start using NetraX."}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-1 rounded-xl bg-[#EAE2D6] p-1 mb-6">
            <button
              type="button"
              onClick={() => switchMode("login")}
              className={`rounded-lg py-2.5 text-sm font-semibold transition ${
                isLogin
                  ? "bg-[#F7F2EB] text-[#2F3728] shadow-sm"
                  : "text-[#69705F] hover:text-[#2F3728]"
              }`}
            >
              Sign In
            </button>

            <button
              type="button"
              onClick={() => switchMode("register")}
              className={`rounded-lg py-2.5 text-sm font-semibold transition ${
                !isLogin
                  ? "bg-[#F7F2EB] text-[#2F3728] shadow-sm"
                  : "text-[#69705F] hover:text-[#2F3728]"
              }`}
            >
              Register
            </button>
          </div>

          {error && (
            <div className="mb-4 rounded-xl border border-[#D5B9B0] bg-[#F2E3DE] px-4 py-3 text-sm text-[#784F43]">
              {error}
            </div>
          )}

          {success && (
            <div className="mb-4 rounded-xl border border-[#C8D0B8] bg-[#EAE2D6] px-4 py-3 text-sm text-[#5D694B]">
              {success}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm font-semibold mb-2">
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
                className="w-full rounded-xl border border-[#D1CBC1] bg-[#F7F2EB] px-4 py-3 text-[#2F3728] outline-none placeholder:text-[#969A90] focus:border-[#8B9A6E] focus:ring-2 focus:ring-[#8B9A6E]/15"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold mb-2">
                Password
              </label>

              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="Enter your password"
                  autoComplete={isLogin ? "current-password" : "new-password"}
                  className="w-full rounded-xl border border-[#D1CBC1] bg-[#F7F2EB] px-4 py-3 pr-12 text-[#2F3728] outline-none placeholder:text-[#969A90] focus:border-[#8B9A6E] focus:ring-2 focus:ring-[#8B9A6E]/15"
                />

                <button
                  type="button"
                  onClick={() => setShowPassword((value) => !value)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-[#69705F] hover:text-[#2F3728]"
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>

              {!isLogin && (
                <p className="text-xs text-[#7C8175] mt-2">
                  Use at least 8 characters.
                </p>
              )}
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-[#8B9A6E] px-4 py-3.5 font-semibold text-[#F7F2EB] hover:bg-[#6F7D55] disabled:opacity-60 disabled:cursor-not-allowed transition"
            >
              {isLogin ? <LogIn size={17} /> : <UserPlus size={17} />}
              {loading
                ? "Please wait..."
                : isLogin
                ? "Sign In"
                : "Create Account"}
            </button>
          </form>

          <div className="mt-6 pt-5 border-t border-[#D8D2C8]">
            <p className="text-xs leading-relaxed text-center text-[#7C8175]">
              NetraX provides AI-assisted analysis for research and
              decision support. Results are not a medical diagnosis.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
