from src.prompts import PromptSampler


def test_prompt_sampler_injects_meta_section():
    sampler = PromptSampler()
    meta_text = "- Explore graph-augmented retrieval\n- Prioritise long-horizon planning"

    prompt = sampler.sample_mutation_prompt(meta_recommendations=meta_text)

    assert "META-ANALYSER RECOMMENDATIONS" in prompt
    assert meta_text in prompt


def test_prompt_sampler_returns_string_without_meta():
    sampler = PromptSampler()

    prompt = sampler.sample_mutation_prompt(meta_recommendations=None)

    assert isinstance(prompt, str)
    assert "META-ANALYSER RECOMMENDATIONS" not in prompt
    assert "### 📋" in prompt
