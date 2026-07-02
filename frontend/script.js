const askBtn = document.getElementById("askBtn");
const question = document.getElementById("question");
const answer = document.getElementById("answer");

const API_URL = "https://unibot-ra-project-production-4823.up.railway.app/query";

askBtn.addEventListener("click", askQuestion);

question.addEventListener("keypress", function(e){
    if(e.key==="Enter"){
        askQuestion();
    }
});

async function askQuestion(){

    const q = question.value.trim();

    if(q===""){
        return;
    }

    answer.innerHTML="<p>Thinking...</p>";

    try{

        const response = await fetch(API_URL,{
            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify({
                query_text:q,
                limit:10
            })
        });

        const data=await response.json();

        if (data.status === "answered") {
    answer.innerHTML = data.answer_text;
} else {
    answer.innerHTML = "<b>No answer found.</b><br>" + (data.answer_text || "");
}

    }catch(err){

        answer.innerHTML="Unable to connect API.";

    }

}