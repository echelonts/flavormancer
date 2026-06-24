// Flavormancer API — minimal skeleton.
// This is the lead-kickoff foundation for M2: a buildable ASP.NET Core app with a
// health check and nothing else. Endpoints, ONNX serving, the rule layer, and the
// molecule service land as their own PRs (see the M2 issues).

var builder = WebApplication.CreateBuilder(args);

var app = builder.Build();

// Liveness probe — confirms the app is up. No business logic here yet.
app.MapGet("/health", () => Results.Ok(new { status = "ok" }));

app.Run();
