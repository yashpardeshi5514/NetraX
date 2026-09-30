import axios from "axios";
import { getAccessToken } from "./auth";

const API_BASE_URL = "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
});

api.interceptors.request.use(
  (config) => {
    const token = getAccessToken();

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error) => Promise.reject(error)
);

export const predictImage = async (file) => {
  const formData = new FormData();

  formData.append("file", file);

  const response = await api.post("/api/predict", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data;
};

export const getAnalysisHistory = async () => {
  const response = await api.get("/api/analyses");

  return response.data;
};

export default api;