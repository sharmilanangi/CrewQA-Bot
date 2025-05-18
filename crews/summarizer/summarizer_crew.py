from crewai import Agent, Task, Crew, Process
from crewai import LLM
from crewai.project import CrewBase, agent, task, crew
from typing import List, Optional
from crewai.agents.agent_builder.base_agent import BaseAgent

@CrewBase
class SummarizerCrew:
    """Crew that generates personalized summaries based on user preferences"""
    
    agents: List[BaseAgent]
    tasks: List[Task]
    
    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or LLM(model="groq/llama-3.3-70b-versatile", temperature=0.1)
    
    @agent
    def summarizer(self) -> Agent:
        """Creates the summarizer agent"""
        return Agent(
            role="Expert Content Summarizer",
            goal="Create personalized summaries tailored to user preferences",
            backstory="""You are an expert at creating summaries that match the 
                        reader's expertise level and desired length. You can break down
                        complex topics for beginners while maintaining technical depth
                        for experts.""",
            llm=self.llm
        )
    
    @task
    def summary_task(self) -> Task:
        """Creates the summary task with user preferences"""
        return Task(
            description="""Analyze and summarize the following content:
                
                Context: {context}
                
                Requirements:
                - Create a {answer_length} length summary
                - Target audience expertise: {expertise_level}
                
                Guidelines:
                - For beginner level: Break down complex concepts, avoid jargon, 
                  provide more context and explanations
                - For intermediate level: Balance technical details with accessibility,
                  include some field-specific terminology
                - For expert level: Use technical language, focus on nuanced details,
                  assume domain knowledge
                
                The summary should be well-structured and maintain the key points
                while matching the user's expertise level.""",
            expected_output="""A well-structured summary that matches the user's 
            expertise level and desired length.""",
            agent=self.summarizer()
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the summarizer crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
