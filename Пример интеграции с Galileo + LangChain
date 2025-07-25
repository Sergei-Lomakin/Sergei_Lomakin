import { OpenAI } from "openai";
import { init, flush, wrapOpenAI } from "galileo";
import dotenv from 'dotenv';

dotenv.config();

// Initialize Galileo
init({
    projectName: "my-project",
    logStreamName: "development"
});

const openai = wrapOpenAI(new OpenAI({ apiKey: process.env.OPENAI_API_KEY }));

async function runAgent() {
    const prompt = "Explain the following topic succinctly: Newton's First Law";
    await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
            { role: "system", content: "You are a helpful assistant." },
            { role: "user", content: prompt }
        ],
        // Дополнительные параметры LLM, если необходимо
    });

    // Flush logs before exiting
    await flush();
}

runAgent();
