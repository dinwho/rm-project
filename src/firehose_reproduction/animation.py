"""Export simulator RGB recordings as a portable side-by-side animation."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def export(folder):
    folder = Path(folder)
    data = json.loads((folder/'data.json').read_text())
    episodes = [next(e for e in data['episodes'] if e['policy'] == name) for name in ('ppo','no_intervention')]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    images=[]
    for i in range(max(len(e['frames']) for e in episodes)):
        image=Image.new('RGB',(900,530),'white');draw=ImageDraw.Draw(image)
        draw.text((20,12),'Original Firehose · published trained policy vs no intervention',font=font,fill='black')
        for k,e in enumerate(episodes):
            frame=e['frames'][min(i,len(e['frames'])-1)]
            x=20+k*450
            grid=Image.fromarray(np.asarray(frame['rgb'],dtype=np.uint8)).resize((400,400),Image.Resampling.NEAREST)
            image.paste(grid,(x,70))
            draw.text((x,44),e['label'],font=font,fill='black')
            draw.text((x,478),f"Decision {frame['step']} · fire-affected {frame['burning']} · treated {frame['harvested']}",font=small,fill='black')
        draw.text((20,508),'Blue: treated · red: upstream fire-affected state · pink: ignition · green/grey: original landscape',font=small,fill='black')
        images.append(image)
    images[0].save(folder/'comparison.gif',save_all=True,append_images=images[1:],duration=180,loop=0)
    images[-1].save(folder/'comparison.png')
    return folder/'comparison.gif'
