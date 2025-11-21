function sendQuestion() {
  const question = document.getElementById("questionInput").value;

  fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById("response").innerText = data.answer;
  });
}
