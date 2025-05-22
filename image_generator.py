from diffusers import StableDiffusionPipeline
import torch
import os

# Ghibli 스타일 모델 로딩
save_dir = "./my_images"
os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성

# Ghibli 스타일 모델 로딩
pipe = StableDiffusionPipeline.from_pretrained(
    "nitrosocke/Ghibli-Diffusion",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# 프롬프트 설정
prompts = [
    "a young girl walking alone through a foggy forest, holding a lantern, surrounded by glowing spirits, feeling curious and calm, in Ghibli style",
    "a boy sitting on a floating dock, dipping his feet into a still lake at sunrise, with mist rising and birds flying overhead, feeling peaceful, in Ghibli style",
    "a girl riding a red bicycle through a sleepy mountain village in early morning light, with dew on the grass and her hair fluttering, feeling free, in Ghibli style",
    "a boy napping on a mossy creature in a forest glade, with warm sunlight filtering through tall trees, feeling safe and warm, in Ghibli style",
    "a girl standing on a windy cliff, watching an airship fly through the clouds, her hair blowing in the wind, eyes full of wonder, in Ghibli style",
    "a child drinking soda beside an old countryside vending machine, fireflies glowing in the evening air, feeling nostalgic, in Ghibli style",
    "a young witch landing on a rural train platform with her broomstick, surrounded by steam and distant mountains, feeling excited and nervous, in Ghibli style",
    "a boy and girl floating in a small wooden boat on a quiet starry lake, glowing flowers drifting on the water, feeling enchanted, in Ghibli style",
    "a girl drawing in a sketchbook under a weeping willow next to a gently flowing stream, feeling calm and focused, in Ghibli style",
    "a boy exploring an overgrown greenhouse filled with wild vines and broken glass, eyes wide with curiosity, in Ghibli style",
    "a girl feeding birds on ancient temple steps at dawn, wrapped in morning mist, feeling serene, in Ghibli style",
    "a boy chasing a paper airplane through golden rice fields under a bright sky, wind in his hair, feeling carefree, in Ghibli style",
    "a girl watching raindrops slide down a train window, the countryside slowly passing by, feeling thoughtful, in Ghibli style",
    "a child curled up with a black cat near a fireplace, while snow falls outside, feeling cozy and protected, in Ghibli style",
    "a young girl floating gently with an umbrella in a soft breeze above rooftops, smiling, feeling dreamy, in Ghibli style",
    "a boy flying a colorful kite on a grassy hill as clouds drift across the sky, laughing, feeling joyful, in Ghibli style",
    "a girl picking wildflowers in a misty meadow at sunrise, surrounded by butterflies, feeling peaceful, in Ghibli style",
    "a child playing a bamboo flute deep in a quiet forest, with woodland animals nearby, feeling magical and connected, in Ghibli style",
    "a boy washing vegetables in a stone sink by a forest stream, birds chirping nearby, feeling mindful, in Ghibli style",
    "a girl hiding under a giant leaf during summer rain, smiling as raindrops fall all around, feeling playful, in Ghibli style",
    "a young boy discovering a glowing egg in the underbrush, watched by forest birds, feeling amazed and cautious, in Ghibli style",
    "a girl climbing a tall tree to reach a secret treehouse, determined and excited, in Ghibli style",
    "a child riding a gentle spirit deer through a twilight valley glowing with light, feeling safe and awestruck, in Ghibli style",
    "a boy helping a fallen scarecrow stand back up in a breezy wheat field, feeling gentle and kind, in Ghibli style",
    "a girl stirring soup in a countryside kitchen filled with sunlight, pots hanging above, feeling warm and nurturing, in Ghibli style",
    "a young witch brewing tea for animals in her cozy wooden cabin, with steam rising and books around her, feeling whimsical, in Ghibli style",
    "a girl painting the sunset sky from a rooftop, surrounded by fluttering birds, feeling inspired and alive, in Ghibli style",
    "a boy sleeping in a hammock under an apple tree, with petals drifting on the breeze, feeling peaceful, in Ghibli style",
    "a girl fishing with her grandfather by the river, both silent and content, in Ghibli style",
    "a child talking to a giant turtle at a calm mountain lake, listening closely, in Ghibli style",
    "a girl walking barefoot through a flower-covered meadow at dusk, with fireflies blinking around, feeling free, in Ghibli style",
    "a boy petting a stray dog in a rainy alleyway, city lights glowing faintly, feeling gentle, in Ghibli style",
    "a young witch lighting lanterns on her balcony at night, stars twinkling above, feeling peaceful, in Ghibli style",
    "a child discovering a hidden garden behind a crumbling stone wall, eyes wide with wonder, in Ghibli style",
    "a girl flying through the sky on a giant paper crane, surrounded by clouds, feeling magical, in Ghibli style",
    "a boy sketching small forest spirits in a worn notebook, sitting on a tree root, in Ghibli style",
    "a girl holding a candle in a vast ancient library with floating books, feeling cautious and amazed, in Ghibli style",
    "a child whispering to a giant cat spirit resting in a wheat field, mysterious and quiet, in Ghibli style",
    "a boy learning to cook from a gentle old spirit in a hidden kitchen, warm and kind, in Ghibli style",
    "a girl planting seeds in a rooftop garden as the sun rises, birds fluttering nearby, hopeful and patient, in Ghibli style",
    "a boy climbing a tree to retrieve a lost hat, laughing as leaves fall around him, in Ghibli style",
    "a girl walking beside a train track during twilight as fireflies glow, feeling dreamy, in Ghibli style",
    "a child and their pet fox stargazing on a hilltop wrapped in a blanket, peaceful and still, in Ghibli style",
    "a girl brushing her hair by a pond while koi fish swim below, morning fog rolling in, in Ghibli style",
    "a boy watching reflections of clouds in a rice paddy, quietly thinking, in Ghibli style",
    "a child riding a wooden boat across a flooded street after heavy rain, observing the silence, in Ghibli style",
    "a girl handing out flowers during a village festival, lanterns floating above, joyful and kind, in Ghibli style",
    "a boy catching falling petals in his hat in a temple garden, softly smiling, in Ghibli style",
    "a girl building a small shrine in the forest with cloth and stones, quietly respectful, in Ghibli style",
    "a child hugging a glowing jellyfish spirit on a moonlit beach, feeling tender and curious, in Ghibli style"
]

# 이미지 여러 장 생성
for i, prompt in enumerate(prompts):
    for j in range(8):
        image = pipe(prompt).images[0]
        filename = f"ghibli_{i+1:03}_{j+1:02}.jpg"
        image.save(os.path.join(save_dir, filename))
