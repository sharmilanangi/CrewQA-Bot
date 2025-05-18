import os
import pathlib
from crewai import LLM
from crewai.flow import Flow, start, listen, router, or_
from crewai.flow.flow import FlowState
from typing import Dict, Any
from crewai import Agent, Task
from crewai import Crew
from crews.summarizer.summarizer_crew import SummarizerCrew
from crews.fact_checker.fact_checker_crew import FactCheckerCrew
from custom_types import UserPreferences

import time
MODEL = "groq/llama-3.3-70b-versatile"

class CrewQAState(FlowState):
    input_query: str = ""
    context: str = ""
    user_preferences: UserPreferences = UserPreferences()
    mode: str = "qa" # can be qa, summarize, fact_check
    claim_to_verify: str = ""
class CrewQAFlow(Flow[CrewQAState]):

    @start()
    def initialize(self):
        """
        Initialize the flow with an LLM
        """
        print("Starting the flow")
        self.llm = LLM(model=MODEL, temperature=0.1)
    
    @router("initialize")
    def set_router(self):
        """
        Set the router to the mode
        """
        print("Setting the router to the mode")
        if self.state.mode == "qa":
            return "qa"
        elif self.state.mode == "summarize":
            return "summarize"
        elif self.state.mode == "fact_check":
            return "fact_check"
        else:
            return "start"
    
    def _update_context_from_transcript(self):
        """
        Update the context from the transcript
        """
        print("Updating the context from the transcript")
        try:
            with open('transcript.txt', 'r') as f:
                self.state.context = f.read()
        except FileNotFoundError:
            print("No transcript.txt file found")
            self.state.context = ""

    @listen("qa")
    def ask_contextual_qa(self):
        """
        Ask the LLM to answer the question
        """
        print("Asking the LLM to answer the question")
        if not self.state.input_query:
            return ""

        self._update_context_from_transcript()
        
        # Time the direct LLM answer
        start_time_llm = time.time()
        response = self._answer()
        llm_time = time.time() - start_time_llm
        print(f"\nDirect LLM Answer (took {llm_time:.2f} seconds):")
        print(response)
        print("---------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # # Create a QA Agent
        # qa_agent = Agent(
        #     role='Expert Question Answerer',
        #     goal='Provide accurate and tailored answers based on context',
        #     backstory="""You are an expert at analyzing context and providing 
        #                 answers that match specific user preferences.""",
        #     llm=self.llm
        # )

        # qa_task = Task(
        #     description=f"""Analyze the following context and answer the question.
        #         Context: {self.state.context}
                
        #         Question: {self.state.input_query}
                
        #         Provide a {self.state.user_preferences.answer_length} length answer 
        #         to an user of {self.state.user_preferences.expertise_level} expertise level in this field. Try to be as informative as possible based on the context and the question.""",
        #     agent=qa_agent,
        #     expected_output="A bullet point list of the answer to the question according to the user preferences."
        # )
        # print("Creating crew and executing task")
        
        # # Time the crew execution
        # start_time_crew = time.time()
        
        # # Create a crew with the agent and task
        # crew = Crew(
        #     agents=[qa_agent],
        #     tasks=[qa_task],
        #     verbose=True
        # )
        
        # # Execute the task using the crew
        # response = crew.kickoff().raw
        # crew_time = time.time() - start_time_crew
        # print(f"Task executed (took {crew_time:.2f} seconds)")
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return response
    
    @listen("summarize")
    def summarizer_module(self):
        """
        Summarize the context
        """
        self._update_context_from_transcript()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(self.state.context)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if self.state.context == "":
            return "Sorry, nothing to summarize yet!"
        summarizer = SummarizerCrew(self.llm)
        summary = summarizer.crew().kickoff(
            inputs = {
                "context": self.state.context,
                "answer_length": self.state.user_preferences.answer_length,
                "expertise_level": self.state.user_preferences.expertise_level
            }
        )
        summary = summary.raw
        return summary
    
    @listen("fact_check")
    def fact_checker_module(self):
        """
        Fact check the context
        """
        self._update_context_from_transcript()
        if self.state.claim_to_verify == "":
            return "Sorry, nothing to fact check yet!"
        fact_checker = FactCheckerCrew()
        response = fact_checker.crew().kickoff(
            inputs = {
                "claim": self.state.claim_to_verify,
                "context": self.state.context
            }
        )
        response = response.raw
        return response

    def _answer(self):
        """Generate a crisp answer based on the context"""
        # Create prompt with instructions for crisp answers
        prompt = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"Context: {self.state.context}\n\n Question: {self.state.input_query}\n\n."}
        ]
        
        # Generate response
        response = self.llm.call(prompt)
        
        # Display the response
        print("\nAnswer:", response)
        return response
    
    def _get_system_prompt(self):
        
        prompt = f"""
        You are a helpful AI assistant that can answer based on the context provided.
        A question is being asked by a user of {self.state.user_preferences.expertise_level} expertise level in this field.
        The answer should be crisp and around {self.state.user_preferences.answer_length} length.
        If no context is provided or if the question is not related to the context, reply with "Sorry, there is no information related to the question in the context provided so far."
        If it is fact based question, provide the answer only if you are sure about the answer from your generic knowledge. If you are not sure, reply with "Sorry, I don't know the answer to that question."
        """

        return prompt

def kickoff():
    flow = CrewQAFlow()
    flow.kickoff()

def plot_flow():
    flow = CrewQAFlow()
    flow.plot()

if __name__ == "__main__":
    plot_flow()
    kickoff()