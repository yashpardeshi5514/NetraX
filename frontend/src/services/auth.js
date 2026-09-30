import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";

const TOKEN_KEY = "netrax_access_token";
const USER_KEY = "netrax_user";

export const registerUser = async (email, password) => {
  const response = await axios.post(
    `${API_BASE_URL}/api/auth/register`,
    {
      email,
      password,
    }
  );

  return response.data;
};

export const loginUser = async (email, password) => {
  const response = await axios.post(
    `${API_BASE_URL}/api/auth/login`,
    {
      email,
      password,
    }
  );

  const { access_token, user } = response.data;

  localStorage.setItem(TOKEN_KEY, access_token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));

  return response.data;
};

export const logoutUser = () => {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
};

export const getAccessToken = () => {
  return localStorage.getItem(TOKEN_KEY);
};

export const getCurrentUser = () => {
  const user = localStorage.getItem(USER_KEY);

  if (!user) {
    return null;
  }

  try {
    return JSON.parse(user);
  } catch {
    return null;
  }
};

export const isAuthenticated = () => {
  return Boolean(getAccessToken());
};