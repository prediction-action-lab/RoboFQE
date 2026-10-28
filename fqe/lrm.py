"""LRM task-completion model: the terminal success detector R̂(o_T, l).

When ground-truth success is unavailable, RoboFQE labels each trajectory with
a terminal success label from a success detector (paper Sec. 3.3-3.4): the
Task Completion model of the Large Reward Model (LRM; Wu et al., 2026,
arXiv:2603.16065), Qwen3-VL-8B-Instruct specialized to answer
`{ "task_completed": "yes" | "no" }` for one observation and a task
instruction (huggingface.co/USC-PSI-Lab/LRM-models, `completion/`). The paper
labels the augmented LIBERO rollouts with LRM fine-tuned on the official
demos (scripts/finetune_lrm.py): off the shelf, LRM does not match LIBERO's
success predicates (e.g. a bowl on the plate counts as success while the
gripper still holds it).

Shared by scripts/label_rewards_lrm.py and scripts/finetune_lrm.py, so that
labeling and fine-tuning see the same prompt, image and answer format. The
prompt and parser follow the LRM model card and its reference reward server
(github.com/physical-superintelligence-lab/Large-Reward-Models). LIBERO
frames are stored in robosuite's OpenGL convention (upside down); "rot180"
gives the orientation of the OpenVLA / LeRobot LIBERO data, the only one in
which LIBERO's "left" / "right" instructions match the image.
"""
import re
from typing import Optional, Tuple

import numpy as np
import torch

DEFAULT_MODEL = "USC-PSI-Lab/LRM-models"
DEFAULT_SUBFOLDER = "completion"
IMAGE_TRANSFORMS = ("rot180", "vflip", "none")

# The completion model's own answer layout (the model card shortens it to one line).
_ANSWER = '{{\n  "task_completed": "{}"\n}}'
_ANSWER_RE = re.compile(r'"task_completed"\s*:\s*"(yes|no)"', re.IGNORECASE)


def default_subfolder(model: str) -> str:
    """`completion/` inside the LRM repo; any other model is a flat directory."""
    return DEFAULT_SUBFOLDER if model == DEFAULT_MODEL else ""


def completion_messages(instruction: str) -> list:
    """The model card's completion query; the image follows its placeholder."""
    return [{"role": "user", "content": [
        {"type": "text", "text": (
            "Task: Determine task completion.\n\n"
            f"The task is: {instruction}\n\n"
            "You are given:\n- Current observation: "
        )},
        {"type": "image"},
        {"type": "text", "text": (
            "\n\nHas the task been completed?\n"
            "Output your answer in the following JSON format:\n"
            '{ "task_completed": "yes" or "no" }'
        )},
    ]}]


def completion_answer(success: bool) -> str:
    """Target response for fine-tuning."""
    return _ANSWER.format("yes" if success else "no")


def parse_completion(response: str) -> Optional[bool]:
    """True / False from the LRM answer; None if it gives no clear yes / no."""
    matches = _ANSWER_RE.findall(response)
    if matches:
        return matches[-1].lower() == "yes"
    # Fallback of the reference server, on whole words.
    has_yes = re.search(r"\byes\b", response, re.IGNORECASE) is not None
    has_no = re.search(r"\bno\b", response, re.IGNORECASE) is not None
    return has_yes if has_yes != has_no else None


def to_pil(frame: np.ndarray, transform: str = "rot180"):
    """(H, W, 3) uint8 LIBERO frame → PIL image in the LRM orientation."""
    from PIL import Image

    if transform == "rot180":
        frame = frame[::-1, ::-1]
    elif transform == "vflip":
        frame = frame[::-1]
    elif transform != "none":
        raise ValueError(f"image transform must be one of {IMAGE_TRANSFORMS}, got {transform!r}")
    return Image.fromarray(np.ascontiguousarray(frame))


def load_lrm(model: str = DEFAULT_MODEL, subfolder: str = DEFAULT_SUBFOLDER,
             adapter: Optional[str] = None, device: str = "cuda"):
    """(model, processor) in bf16 on `device`, eval mode.

    `adapter`: a LoRA from scripts/finetune_lrm.py, merged into the weights.
    It is attached on CPU before the move: on a device_map-dispatched model
    PEFT can silently keep lora_B at its zero init (see the LRM server).
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kwargs = {"subfolder": subfolder} if subfolder else {}
    processor = AutoProcessor.from_pretrained(model, **kwargs)
    net = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16, **kwargs)
    if adapter:
        from peft import PeftModel

        net = PeftModel.from_pretrained(net, adapter)
        if not any(p.abs().sum() > 0 for n, p in net.named_parameters() if "lora_B" in n):
            raise RuntimeError(f"adapter {adapter}: all lora_B weights are zero (not loaded)")
        net = net.merge_and_unload()
    return net.to(device).eval(), processor


def completion_inputs(processor, image, instruction: str, answer: Optional[str] = None):
    """Model inputs of one completion query.

    With `answer`, its tokens and <|im_end|> are appended as the assistant
    turn and `labels` keep only them (prompt masked with -100). They are
    appended as token ids, as generation produces them after the prompt.
    """
    text = processor.apply_chat_template(
        completion_messages(instruction), tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    if answer is not None:
        tok = processor.tokenizer
        ids = tok(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
        ids = torch.cat([ids, torch.tensor([[tok.convert_tokens_to_ids("<|im_end|>")]])], dim=1)
        n_prompt = inputs["input_ids"].shape[1]
        inputs["input_ids"] = torch.cat([inputs["input_ids"], ids], dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        if "mm_token_type_ids" in inputs:  # 0 = text token
            inputs["mm_token_type_ids"] = torch.cat(
                [inputs["mm_token_type_ids"], torch.zeros_like(ids)], dim=1
            )
        labels = inputs["input_ids"].clone()
        labels[:, :n_prompt] = -100
        inputs["labels"] = labels
    return inputs


@torch.no_grad()
def predict(model, processor, image, instruction: str,
            max_new_tokens: int = 32) -> Tuple[Optional[bool], str]:
    """Greedy LRM answer → (completed, or None if unparseable; raw response)."""
    inputs = completion_inputs(processor, image, instruction).to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=None, top_p=None, top_k=None,
    )
    response = processor.tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return parse_completion(response), response
