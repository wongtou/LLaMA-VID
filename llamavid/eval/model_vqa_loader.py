
# 导入必要的库
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid


# 导入项目相关模块
from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math



def split_list(lst, n):
    """
    将列表 lst 均匀分成 n 个块。
    Args:
        lst: 待分割的列表
        n: 块数
    Returns:
        List[List]: 分割后的列表块
    """
    chunk_size = math.ceil(len(lst) / n)  # 向上取整，保证分块均匀
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]



def get_chunk(lst, n, k):
    """
    获取第 k 个块。
    Args:
        lst: 待分割的列表
        n: 块数
        k: 块索引
    Returns:
        List: 第 k 个块
    """
    chunks = split_list(lst, n)
    return chunks[k]



# 自定义数据集类，用于加载问题和图片
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        """
        Args:
            questions: 问题列表，每个元素为 dict，包含 image 和 text 字段
            image_folder: 图片文件夹路径
            tokenizer: 文本分词器
            image_processor: 图片预处理器
            model_config: 模型配置
        """
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        """
        获取单条数据，包括输入 token ids 和图片 tensor。
        """
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        # 构造带图片 token 的 prompt
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # 构造对话模板
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 加载图片并处理
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        # 对 prompt 进行分词和编码
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        """
        返回数据集长度
        """
        return len(self.questions)



# 构建 DataLoader，批量加载数据
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    """
    创建用于推理的数据加载器。
    Args:
        questions: 问题列表
        image_folder: 图片文件夹
        tokenizer: 分词器
        image_processor: 图片预处理器
        model_config: 模型配置
        batch_size: 批大小（必须为1）
        num_workers: 工作线程数
    Returns:
        DataLoader: pytorch 数据加载器
    """
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader



def eval_model(args):
    """
    主评测函数，加载模型、数据，推理并保存结果。
    Args:
        args: 命令行参数
    """
    # 禁用 torch 默认初始化，提升推理速度
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # 加载预训练模型和相关处理器
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    # 加载问题数据，并分块处理（支持多卡/多进程分块）
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # 自动切换对话模式（plain 模型自动加 mmtag）
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # 构建数据加载器
    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    # 推理主循环
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        # 获取停止符
        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)        
        model.update_prompt([[cur_prompt]])

        # 推理生成答案
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)

        # 检查输入输出 token 差异
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # 解码输出文本
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # 保存结果到文件
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()


# 主入口，解析命令行参数并启动评测
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m", help="模型路径")
    parser.add_argument("--model-base", type=str, default=None, help="基础模型路径")
    parser.add_argument("--image-folder", type=str, default="", help="图片文件夹路径")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl", help="问题文件路径")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl", help="答案输出文件路径")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="对话模式")
    parser.add_argument("--num-chunks", type=int, default=1, help="分块数量")
    parser.add_argument("--chunk-idx", type=int, default=0, help="当前块索引")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度")
    parser.add_argument("--top_p", type=float, default=None, help="top_p 策略")
    parser.add_argument("--num_beams", type=int, default=1, help="beam search 数量")
    args = parser.parse_args()

    eval_model(args)
