from .base_agent import BaseAgent
from prompts import (
    exposition_writing_prompt,
    rising_action_writing_prompt,
    climax_writing_prompt,
    falling_action_writing_prompt,
    resolution_writing_prompt
)

class WritingAgent(BaseAgent):
    """
    A self-contained agent that takes a completed plan and writes a full story,
    section by section, following a narrative arc.
    """
    def __init__(self, *args, **kwargs):
        """Initializes the WritingAgent by inheriting from BaseAgent."""
        super().__init__(*args, **kwargs)

    def write_story(self, idx):
        """
        Takes a plan scratchpad and writes a full story by sequentially writing
        """
        print(f" Starting story writing pipeline for task index {idx}...")

        # The scratchpad starts with the full plan from the PlanningAgent.
        final_story = []

        # --- Step 1: Write Exposition ---
        print("  - Writing [Exposition]...")
        exposition_result = self._write_exposition(self, idx)
        exposition_text = exposition_result['unwrapped_text']
        self.scratchpad += f"\n\n[Exposition]\n{exposition_text}"
        final_story.append(exposition_text)

        # --- Step 2: Write Rising Action ---
        print("  - Writing [Rising Action]...")
        rising_action_result = self._write_rising_action(self, idx)
        rising_action_text = rising_action_result['unwrapped_text']
        scratchpad += f"\n\n[Rising Action]\n{rising_action_text}"
        final_story.append(rising_action_text)

        # --- Step 3: Write Climax ---
        print("  - Writing [Climax]...")
        climax_result = self._write_climax(self, idx)
        climax_text = climax_result['unwrapped_text']
        scratchpad += f"\n\n[Climax]\n{climax_text}"
        final_story.append(climax_text)

        # --- Step 4: Write Falling Action ---
        print("  - Writing [Falling Action]...")
        falling_action_result = self._write_falling_action(self, idx)
        falling_action_text = falling_action_result['unwrapped_text']
        scratchpad += f"\n\n[Falling Action]\n{falling_action_text}"
        final_story.append(falling_action_text)

        # --- Step 5: Write Resolution ---
        print("  - Writing [Resolution]...")
        resolution_result = self._write_resolution(self, idx)
        resolution_text = resolution_result['unwrapped_text']
        # No need to add the final part to the scratchpad if it's the end.
        final_story.append(resolution_text)

        print("✅ Story writing complete.")
        # Join the parts of the story into a single string.
        return "\n\n".join(final_story)

    # --- Private methods for each writing step ---

    def _write_exposition(self, idx):
        prompt = exposition_writing_prompt.format(section="Exposition", scratchpad=self.scratchpad)
        return self.process_single_instance(
            i=idx, method="write", prompt=prompt, test_output=False, phase="exposition"
        )

    def _write_rising_action(self, idx):
        prompt = rising_action_writing_prompt.format(section="Rising Action", scratchpad=self.scratchpad)
        return self.process_single_instance(
            i=idx, method="write", prompt=prompt, test_output=False, phase="rising_action"
        )

    def _write_climax(self, idx):
        prompt = climax_writing_prompt.format(section="Climax", scratchpad=self.scratchpad)
        return self.process_single_instance(
            i=idx, method="write", prompt=prompt, test_output=False, phase="climax"
        )

    def _write_falling_action(self, idx):
        prompt = falling_action_writing_prompt.format(section="Falling Action", scratchpad=self.scratchpad)
        return self.process_single_instance(
            i=idx, method="write", prompt=prompt, test_output=False, phase="falling_action"
        )

    def _write_resolution(self, idx):
        prompt = resolution_writing_prompt.format(section="Resolution", scratchpad=self.scratchpad)
        return self.process_single_instance(
            i=idx, method="write", prompt=prompt, test_output=False, phase="resolution"
        )
