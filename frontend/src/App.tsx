
import { useEffect, useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8001";
type DocumentItem = {
  id: string;
  filename: string;
  file_type: string;
  size_bytes: number;
  created_at: string;
  text_characters: number;
};

type DocumentDetail = DocumentItem & {
  extracted_text: string;
};
type SearchResult = {
  document_id: string;
  filename: string;
  chunk_index: number;
  text: string;
  cosine_distance: number;
  similarity_score: number;
};

type SearchResponse = {
  query: string;
  count: number;
  results: SearchResult[];
};
type AskSource = {
  document_id: string;
  filename: string;
  chunk_index: number;
  text: string;
  similarity_score: number;
};

type AskResponse = {
  question: string;
  answer: string;
  sources: AskSource[];
};

function App() {
  const [chatQuestion, setChatQuestion] = useState("");
const [asking, setAsking] = useState(false);
const [chatError, setChatError] = useState("");
const [chatResponse, setChatResponse] = useState<AskResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [selected, setSelected] = useState<DocumentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [query, setQuery] = useState("");
const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
const [searching, setSearching] = useState(false);
const [searchMessage, setSearchMessage] = useState("");

  async function loadDocuments() {
    try {
      const response = await fetch(`${API_URL}/documents`);
      if (!response.ok) throw new Error("Could not load documents.");

      const data: DocumentItem[] = await response.json();
      setDocuments(data);
      setError("");
    } catch {
      setError("Could not connect to the API. Check that FastAPI is running.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadDocuments();
  }, []);

  async function handleUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setError("");
    setMessage("");

    const allowed = [".pdf", ".txt", ".md"];
    const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();

    if (!allowed.includes(extension)) {
      setError("Choose a PDF, TXT, or Markdown file.");
      event.target.value = "";
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("File size must be 10 MB or less.");
      event.target.value = "";
      return;
    }

    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/documents`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Upload failed.");
      }

      setMessage(`Successfully uploaded ${result.filename}.`);
      await loadDocuments();

      const detailResponse = await fetch(
        `${API_URL}/documents/${result.id}`,
      );

      if (detailResponse.ok) {
        setSelected(await detailResponse.json());
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "An unexpected error occurred.",
      );
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function openDocument(id: string) {
    setError("");

    try {
      const response = await fetch(`${API_URL}/documents/${id}`);
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Could not open document.");
      }

      setSelected(result);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not open document.",
      );
    }
  }
async function handleSearch(event: React.FormEvent<HTMLFormElement>) {
  event.preventDefault();

  const trimmedQuery = query.trim();

  if (!trimmedQuery) {
    setSearchMessage("Enter a question to search your knowledge base.");
    setSearchResults([]);
    return;
  }

  setSearching(true);
  setSearchMessage("");
  setError("");
  setSelected(null);

  try {
    const response = await fetch(`${API_URL}/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: trimmedQuery,
        limit: 5,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Search failed.");
    }

    const result: SearchResponse = data;
    setSearchResults(result.results);

    if (result.count === 0) {
      setSearchMessage("No matching document chunks found.");
    }
  } catch (err) {
    setSearchMessage(
      err instanceof Error ? err.message : "Could not search documents.",
    );
  } finally {
    setSearching(false);
  }
}
async function handleAsk(event: React.FormEvent<HTMLFormElement>) {
  event.preventDefault();

  const question = chatQuestion.trim();

  if (!question) {
    setChatError("Please enter a question.");
    return;
  }

  setAsking(true);
  setChatError("");
  setChatResponse(null);

  try {
    const response = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        limit: 5,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Could not generate an answer.");
    }

    setChatResponse(data as AskResponse);
  } catch (err) {
    setChatError(
      err instanceof Error
        ? err.message
        : "Could not connect to the AI service."
    );
  } finally {
    setAsking(false);
  }
}

  function formatSize(bytes: number) {
    return bytes < 1024
      ? `${bytes} B`
      : `${(bytes / 1024).toFixed(1)} KB`;
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-icon">S</span>
          <span>Second Brain</span>
        </div>

        <p className="nav-label">WORKSPACE</p>
        <div className="nav-item active">▤ &nbsp; My knowledge</div>
        <div className="sidebar-note">
          Your personal library for documents, ideas, and research.
        </div>

        <div className="sidebar-bottom">SECOND BRAIN OS · v0.2.0</div>
      </aside>

      <section className="main-content">
        <header className="topbar">
          <span>My workspace / Knowledge library</span>
          <span className="api-status">
            <span className="status-dot" />
            API connected
          </span>
        </header>

        <div className="page-content">
          <section className="page-heading">
            <div>
              <p className="eyebrow">YOUR PERSONAL KNOWLEDGE BASE</p>
              <h1>My knowledge</h1>
              <p className="subtitle">
                Bring your information together. Find it when you need it.
              </p>
            </div>
            <label className={`upload-button ${uploading ? "disabled" : ""}`}>
              {uploading ? "Uploading..." : "+ Upload document"}
              <input
                type="file"
                accept=".pdf,.txt,.md"
                onChange={handleUpload}
                disabled={uploading}
                hidden
              />
            </label>
          </section>
          
<section className="library-panel search-panel">
  <div className="panel-heading">
    <div>
      <p className="eyebrow">AI-POWERED RETRIEVAL</p>
      <h2>Search your knowledge</h2>
      <p>Ask a question to find relevant passages in your documents.</p>
    </div>
  </div>

  <form className="search-form" onSubmit={handleSearch}>
    <input
      className="search-input"
      type="text"
      value={query}
      onChange={(event) => setQuery(event.target.value)}
      placeholder="Ask something, e.g. Who is Rahul?"
      maxLength={2000}
      aria-label="Search your knowledge"
    />
    <button
      className="upload-button"
      type="submit"
      disabled={searching}
    >
      {searching ? "Searching..." : "Search"}
    </button>
  </form>

  {searchMessage && (
    <p className="search-message" role="status">
      {searchMessage}
    </p>
  )}

  {searchResults.length > 0 && (
    <div className="search-results">
      <h3>Relevant passages ({searchResults.length})</h3>

      {searchResults.map((result, index) => (
        <article
          className="search-result"
          key={`${result.document_id}-${result.chunk_index}`}
        >
          <div className="search-result-heading">
            <strong>{index + 1}. {result.filename}</strong>
            <span>
              Distance: {result.cosine_distance.toFixed(4)}
            </span>
          </div>

          <p>{result.text}</p>

          <button
            className="refresh-button"
            type="button"
            onClick={() => void openDocument(result.document_id)}
          >
            Open original document
          </button>
        </article>
      ))}
    </div>
  )}
</section>


<section className="library-panel chat-panel">
  <div className="panel-heading">
    <div>
      <p className="eyebrow">RETRIEVAL-AUGMENTED GENERATION</p>
      <h2>Chat with your knowledge</h2>
      <p>Ask questions and get answers grounded in your documents.</p>
    </div>
    <span className="chat-badge">AI · RAG</span>
  </div>

  <form className="chat-form" onSubmit={handleAsk}>
    <textarea
      className="chat-input"
      value={chatQuestion}
      onChange={(event) => setChatQuestion(event.target.value)}
      placeholder="Ask your documents anything..."
      maxLength={2000}
      rows={3}
      aria-label="Question for AI"
      disabled={asking}
    />

    <div className="chat-form-footer">
      <span>Answers include supporting document sources.</span>
      <button
        className="upload-button"
        type="submit"
        disabled={asking || !chatQuestion.trim()}
      >
        {asking ? "Thinking..." : "Ask AI ↗"}
      </button>
    </div>
  </form>

  {asking && (
    <div className="chat-loading" role="status">
      <span className="loading-dot" />
      Searching your knowledge and generating an answer...
    </div>
  )}

  {chatError && (
    <div className="notice error" role="alert">
      {chatError}
    </div>
  )}

  {chatResponse && (
    <div className="chat-answer">
      <p className="eyebrow">ANSWER</p>
      <h3>{chatResponse.question}</h3>
      <div className="answer-text">{chatResponse.answer}</div>

      <div className="source-section">
        <h3>Sources ({chatResponse.sources.length})</h3>

        {chatResponse.sources.length === 0 ? (
          <p className="source-empty">
            No indexed sources were returned for this answer.
          </p>
        ) : (
          chatResponse.sources.map((source) => (
            <details
              className="source-card"
              key={`${source.document_id}-${source.chunk_index}`}
            >
              <summary>
                <span className="source-file-icon">DOC</span>
                <span className="source-file-name">{source.filename}</span>
                <span className="source-score">
                  {(source.similarity_score * 100).toFixed(1)}%
                </span>
              </summary>

              <p>{source.text}</p>

              <button
                className="refresh-button"
                type="button"
                onClick={() => void openDocument(source.document_id)}
              >
                Open original document
              </button>
            </details>
          ))
        )}
      </div>
    </div>
  )}
</section>
 

          {error && <div className="notice error">{error}</div>}
          {message && <div className="notice success">{message}</div>}

          <section className="stats-grid">
            <div className="stat-card">
              <span className="stat-label">TOTAL DOCUMENTS</span>
              <strong>{documents.length}</strong>
              <span className="stat-foot">In your library</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">SUPPORTED FORMATS</span>
              <strong>3</strong>
              <span className="stat-foot">PDF, TXT, Markdown</span>
            </div>
          </section>

          <section className="library-panel">
            <div className="panel-heading">
              <div>
                <h2>Your documents</h2>
                <p>Files and extracted text stored in your knowledge base.</p>
              </div>
              <button className="refresh-button" onClick={() => void loadDocuments()}>
                Refresh
              </button>
            </div>

            {loading ? (
              <p className="empty-state">Loading documents...</p>
            ) : documents.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">▤</div>
                <h3>Your library is waiting</h3>
                <p>Upload your first document to get started.</p>
              </div>
            ) : (
              <div className="document-list">
                {documents.map((doc) => (
                  <button
                    className={`document-row ${
                      selected?.id === doc.id ? "selected" : ""
                    }`}
                    key={doc.id}
                    onClick={() => void openDocument(doc.id)}
                  >
                    <span className="file-icon">
                      {doc.file_type === "pdf" ? "PDF" : "TXT"}
                    </span>
                    <span className="document-info">
                      <strong>{doc.filename}</strong>
                      <small>
                        {formatSize(doc.size_bytes)} ·{" "}
                        {doc.text_characters.toLocaleString()} characters
                      </small>
                    </span>
                    <span className="document-date">
                      {new Date(doc.created_at).toLocaleDateString()}
                    </span>
                    <span className="open-arrow">›</span>
                  </button>
                ))}
              </div>
            )}
          </section>

          {selected && (
            <section className="preview-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">DOCUMENT PREVIEW</p>
                  <h2>{selected.filename}</h2>
                </div>
                <button
                  className="refresh-button"
                  onClick={() => setSelected(null)}
                >
                  Close
                </button>
              </div>
              <p className="preview-meta">
                {selected.file_type.toUpperCase()} ·{" "}
                {(selected.extracted_text ?? "").length.toLocaleString()} characters
              </p>
              <pre className="extracted-text">{selected.extracted_text}</pre>
            </section>
          )}

          <footer className="page-footer">
            Second Brain OS · Your knowledge, organized.
          </footer>
        </div>
      </section>
    </main>
  );
}

export default App;
