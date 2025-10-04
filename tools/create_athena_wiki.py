#!/usr/bin/env python3
"""
Athena Wiki Creator - Multi-step research and wiki generation using LiteLLM models.
"""

import os
import re
from datetime import datetime
from litellm_wrapper import LiteLLMChat
from semantic_agent import SemanticAgent

class AthenaWikiCreator:
    def __init__(self):
        self.claude = LiteLLMChat("claude-sonnet-4-with-reasoning")
        self.semantic_agent = SemanticAgent()  # Replace rag-athena-adv with graph RAG
        self.gpt5 = LiteLLMChat("gpt-5-chat")
        
    def make_llm_call(self, model, system_prompt, user_prompt):
        """Make a call to any LiteLLM model using the wrapper"""
        try:
            # Combine system and user prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            if model == "claude-sonnet-4-with-reasoning":
                return self.claude.invoke(full_prompt)
            elif model == "rag-athena-adv":
                # Use semantic agent for graph RAG instead of rag-athena-adv
                return self.semantic_agent.run_query(full_prompt)
            elif model == "gpt-5-chat":
                return self.gpt5.invoke(full_prompt)
            else:
                llm = LiteLLMChat(model)
                return llm.invoke(full_prompt)
            
        except Exception as e:
            print(f"Error calling {model}: {e}")
            return None
    
    def step1_research_planning(self, research_topic):
        """Step 1: Use claude-sonnet-4-with-reasoning to create research plan"""
        print("🔍 Step 1: Creating research plan...")
        
        system_prompt = """You are a research planning assistant that takes a question or topic and then builds a plan for an agent that will query Ubisoft video game company's internal knowledgebase to generate a wiki on this topic. You prepare a detailed plan of questions to ask this model and respond with three ### separated questions to ask the knowledgebase, and nothing else.

Example: for research topic "Assassin's Creed Shadows performance", you reply with:
###How was the critical response to Assassin's Creed Shadows?
###What was the commercial reception of Assassin's Creed Shadows?
###What was the player response to Assassin's Creed Shadows?"""

        user_prompt = f"Research topic: {research_topic}"
        
        response = self.make_llm_call("claude-sonnet-4-with-reasoning", system_prompt, user_prompt)
        
        if response:
            # Split the questions
            questions = [q.strip() for q in response.split("###") if q.strip()]
            print(f"Generated {len(questions)} research questions")
            return questions
        return []
    
    def step2_knowledgebase_query(self, questions):
        """Step 2: Query rag-athena-adv with the research questions"""
        print("📚 Step 2: Querying knowledgebase...")
        
        answers = []
        for i, question in enumerate(questions, 1):
            print(f"  Querying question {i}: {question[:50]}...")
            
            response = self.make_llm_call("rag-athena-adv", "", question)
            if response:
                answers.append(response)
            else:
                answers.append("No response received")
        
        return answers
    
    def step3_followup_planning(self, original_topic, initial_questions, initial_answers):
        """Step 3: Generate follow-up questions using claude-sonnet-4-with-reasoning"""
        print("🔄 Step 3: Planning follow-up research...")
        
        system_prompt = """You are a research planning assistant. Based on the original question, follow-up questions, and their responses, you need to further interrogate Ubisoft's knowledgebase for follow-up questions. Respond with three ### separated questions, and nothing else.

Example follow-up questions:
###What were the technical challenges mentioned in the responses?
###How did the development team address the issues raised?
###What were the long-term impacts mentioned?"""

        user_prompt = f"""This was the original question: {original_topic}

These were the follow-up questions: {initial_questions}

These were the responses: {initial_answers}

Now your job is to further interrogate Ubisoft's knowledgebase for follow-up questions and respond with three ### separated questions."""
        
        response = self.make_llm_call("claude-sonnet-4-with-reasoning", system_prompt, user_prompt)
        
        if response:
            followup_questions = [q.strip() for q in response.split("###") if q.strip()]
            print(f"Generated {len(followup_questions)} follow-up questions")
            return followup_questions
        return []
    
    def step4_followup_queries(self, followup_questions):
        """Step 4: Query rag-athena-adv with follow-up questions"""
        print("🔍 Step 4: Querying follow-up questions...")
        
        followup_answers = []
        for i, question in enumerate(followup_questions, 1):
            print(f"  Querying follow-up {i}: {question[:50]}...")
            
            response = self.make_llm_call("rag-athena-adv", "", question)
            if response:
                followup_answers.append(response)
            else:
                followup_answers.append("No response received")
        
        return followup_answers
    
    def step5_wiki_generation(self, original_topic, initial_questions, initial_answers, followup_questions, followup_answers):
        """Step 5: Generate final wiki using gpt-5-chat"""
        print("📝 Step 5: Generating wiki...")
        
        system_prompt = f"""You are a wiki generation assistant that creates a structured wiki on the topic "{original_topic}" with the information provided. Create a comprehensive, well-structured markdown wiki with proper headings, sections, and formatting.

IMPORTANT: Do not ask any follow-up questions or suggest additional research. Simply provide the complete wiki content based on the information given. End your response with the final wiki content only."""
        
        user_prompt = f"""Original research topic: {original_topic}

Initial research questions:
{initial_questions}

Initial research answers:
{initial_answers}

Follow-up questions:
{followup_questions}

Follow-up answers:
{followup_answers}

Please create a comprehensive wiki article using all this information. Do not ask any follow-up questions or suggest additional research. Provide only the complete wiki content.

At the end of the wiki, add a "Research Methodology" section that lists:
- All initial research questions asked
- All follow-up questions asked
- Brief description of the research process

End your response with the final wiki content only. Do not include any questions or suggestions for further research."""
        
        response = self.make_llm_call("gpt-5-chat", system_prompt, user_prompt)
        return response
    
    def save_wiki(self, content, topic):
        """Save wiki to Wikis/ folder with GPT-generated filename"""
        # Create Wikis directory if it doesn't exist
        os.makedirs("Wikis", exist_ok=True)
        
        # Generate a better filename using GPT-4o-mini
        print("📝 Generating wiki filename...")
        filename_prompt = f"""Based on this research topic: "{topic}"

Generate a short, descriptive filename for a wiki article (without .md extension). 
The filename should be:
- Short and concise (max 50 characters)
- Descriptive of the content
- Use underscores instead of spaces
- Avoid special characters
- Be professional and clear

Examples: "AC_Shadows_vs_Valhalla_Comparison", "Game_Performance_Analysis", "Player_Engagement_Study"

Return only the filename, nothing else."""
        
        try:
            generated_name = self.make_llm_call("gpt-4o-mini", "", filename_prompt)
            if generated_name and len(generated_name.strip()) > 0:
                # Clean the generated name
                clean_name = re.sub(r'[^\w\s-]', '', generated_name.strip())
                clean_name = re.sub(r'[-\s]+', '_', clean_name)
                filename = f"Wikis/{clean_name}.md"
            else:
                # Fallback to original method if GPT fails
                safe_topic = re.sub(r'[^\w\s-]', '', topic).strip()
                safe_topic = re.sub(r'[-\s]+', '_', safe_topic)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"Wikis/wiki_{safe_topic}_{timestamp}.md"
        except Exception as e:
            print(f"⚠️ Error generating filename with GPT: {e}")
            # Fallback to original method
            safe_topic = re.sub(r'[^\w\s-]', '', topic).strip()
            safe_topic = re.sub(r'[-\s]+', '_', safe_topic)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Wikis/wiki_{safe_topic}_{timestamp}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Wiki saved to: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving wiki: {e}")
            return None
    
    def create_wiki(self, research_topic):
        """Main method to create wiki following the 4-step process"""
        print(f"🚀 Starting Athena Wiki Creation for: {research_topic}")
        print("=" * 60)
        
        # Step 1: Research planning
        initial_questions = self.step1_research_planning(research_topic)
        if not initial_questions:
            print("❌ Failed to generate initial research questions")
            return None
        
        # Step 2: Query knowledgebase
        initial_answers = self.step2_knowledgebase_query(initial_questions)
        if not initial_answers:
            print("❌ Failed to get initial answers")
            return None
        
        # Step 3: Follow-up planning
        followup_questions = self.step3_followup_planning(research_topic, initial_questions, initial_answers)
        if not followup_questions:
            print("❌ Failed to generate follow-up questions")
            return None
        
        # Step 4: Follow-up queries
        followup_answers = self.step4_followup_queries(followup_questions)
        if not followup_answers:
            print("❌ Failed to get follow-up answers")
            return None
        
        # Step 5: Wiki generation
        wiki_content = self.step5_wiki_generation(
            research_topic, 
            initial_questions, 
            initial_answers, 
            followup_questions, 
            followup_answers
        )
        
        if not wiki_content:
            print("❌ Failed to generate wiki content")
            return None
        
        # Save wiki
        filename = self.save_wiki(wiki_content, research_topic)
        
        print(f"\n🎉 Wiki creation completed!")
        print(f"📄 Generated {len(wiki_content)} characters")
        print(f"💾 Saved to: {filename}")
        
        return filename

def main():
    """Main function"""
    creator = AthenaWikiCreator()
    
    # Get research topic from user
    research_topic = input("Enter your research topic: ").strip()
    
    if not research_topic:
        print("❌ No research topic provided")
        return
    
    # Create wiki
    result = creator.create_wiki(research_topic)
    
    if result:
        print(f"\n✅ Success! Wiki created at: {result}")
    else:
        print("\n❌ Wiki creation failed")

if __name__ == "__main__":
    main()