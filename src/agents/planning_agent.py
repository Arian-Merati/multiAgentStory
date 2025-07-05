from .base_agent import BaseAgent
from prompts import *


class PlanningAgent(BaseAgent):
    def __init__(self, model, processor, device):
        super().__init__(model, processor, device)
    
    def plan(self, i):
        """
        Plan the output of a single instance using Agent room planning.
        """
        print("\tidx:", i, "start planning...")
        
        initial_task_prompt = self.task.get_input_prompt(i, method="standard")
        scratchpad = f"[Creative Writing Task]\n{initial_task_prompt}"
        
        print("  - Generating [Character Descriptions]...")
        character_result = self._generate_characters(i, scratchpad)
        scratchpad += f"\n\n[Character Descriptions]\n{character_result['unwrapped_text']}"
        
        print("  - Generating [Central Conflict]...")
        conflict_result = self._generate_conflict(i, scratchpad)
        scratchpad += f"\n\n[Central Conflict]\n{conflict_result['unwrapped_text']}"

        print("  - Generating [Setting]...")
        setting_result = self._generate_setting(i, scratchpad)
        scratchpad += f"\n\n[Setting]\n{setting_result['unwrapped_text']}"

        print("  - Generating [Key Plot Points]...")
        plot_result = self._generate_plot(i, scratchpad)
        scratchpad += f"\n\n[Key Plot Points]\n{plot_result['unwrapped_text']}"

        print(" Planning complete.")
        return scratchpad

    def _generate_conflict(self, idx: int, current_scratchpad: str):
        prompt = conflict_plan_prompt.format(scratchpad=current_scratchpad)
        return self.process_single_instance(
            i=idx, method="plan", prompt=prompt, test_output=False, phase="conflict"
        )

    def _generate_characters(self, idx: int, current_scratchpad: str):
        prompt = character_plan_prompt.format(scratchpad=current_scratchpad)
        return self.process_single_instance(
            i=idx, method="plan", prompt=prompt, test_output=False, phase="characters"
        )

    def _generate_setting(self, idx: int, current_scratchpad: str):
        prompt = setting_plan_prompt.format(scratchpad=current_scratchpad)
        return self.process_single_instance(
            i=idx, method="plan", prompt=prompt, test_output=False, phase="setting"
        )

    def _generate_plot(self, idx: int, current_scratchpad: str):
        prompt = plot_plan_prompt.format(scratchpad=current_scratchpad)
        return self.process_single_instance(
            i=idx, method="plan", prompt=prompt, test_output=False, phase="plot"
        )

        
        