import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "@fontsource-variable/space-grotesk";
import "@xyflow/react/dist/style.css";
import "./index.css";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { Toaster } from "sonner";
import App from "./App";
import { DocProvider } from "./components/domain/DocViewer";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 10_000, retry: 1, refetchOnWindowFocus: false } },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <DocProvider>
          <App />
        </DocProvider>
        <Toaster richColors closeButton position="bottom-right" theme={document.documentElement.classList.contains("dark") ? "dark" : "light"} />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
