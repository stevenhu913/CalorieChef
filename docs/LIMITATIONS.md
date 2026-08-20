# Limitations

- CalorieChef provides educational nutrition information, not medical advice,
  diagnosis, treatment, or guaranteed health outcomes.
- USDA records vary by preparation, product, and serving basis; candidate
  selection may require clarification.
- The application has no authentication, authorization, or rate limiting.
- SQLite state is local to one running instance and is not a production
  multi-user persistence guarantee.
- Hosted deployment has no durable cloud memory or hosted vector store.
- Local long-term memory requires Chroma and a compatible Ollama embedding
  model.
- Language-model tool adherence and structured-output reliability can vary.
- The optional semantic Judge may be unstable and can share correlated errors
  with the Agent model.
- The Multi-Agent experiment did not demonstrate a grounded nutrition happy
  path or a repeatable quality improvement. Its single controlled observation
  is not a statistical benchmark.
- No public deployment is claimed until a real URL passes the documented
  acceptance checks.
