# src/guide_creator_flow/crews/content_crew/content_crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool
from crewai import LLM

@CrewBase
class FactCheckerCrew():
    """Fact checking crew that verifies claims using research and analysis"""

    agents: List[BaseAgent]
    tasks: List[Task]
    # llm = LLM(model="groq/llama-3.3-70b-versatile")
    # llm =LLM( model="gemini/gemini-2.0-flash",
    #     temperature=0.7,
    # )
    llm = LLM(
        model="gemini/gemini-2.0-flash-lite",
        temperature=0.7,
    )

    @agent
    def researcher(self) -> Agent:
        search_tool = SerperDevTool()
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            tools=[search_tool],
            verbose=True,
            llm=self.llm
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['analyst'], # type: ignore[index]
            verbose=True,
            llm=self.llm
        )

    @task
    def claim_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['claim_research_task'] # type: ignore[index]
        )

    @task
    def claim_verification_task(self) -> Task:
        return Task(
            config=self.tasks_config['claim_verification_task'], # type: ignore[index]
            context=[self.claim_research_task()]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the fact checking crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
