from pathlib import Path
from pathlib import Path
from tkinter import *
from argparse import ArgumentParser
from PIL import Image, ImageTk

Image.DEBUG = 0


class Config:

    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for creating gifs of input videos.')
        self.parser.add_argument('--input_path', type=str, default="merges/", help="Path to data directory.")
        self.parser.add_argument('--start', type=int, default=0, help="Starting lane change clip ex: for 10410_201706081335 use 10410")
        self.parser.add_argument('--frame_delay', type=int, default=1, help="The amount of delay (ms) between each clip")
        self.parser.add_argument('--filter', type=str, default='', help="Type of maneuver to consider {branches, lanechange, merges, turns2}")
        self.parser.add_argument('--risk', default=False, action='store_true', help="Set to show only risky clips")
        self.parser.add_argument('--nonrisk', default=False, action='store_true', help="Set to show only non-risky clips")
        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        self.input_base_dir = Path(self.input_path).resolve()

def filterClip(foldername):
    ignore_file = foldername / "ignore.txt"
    if ignore_file.exists():
        with open(str(ignore_file), 'r') as f:
            ignore_data = int(f.read())
            if ignore_data:
                return 1
            else:
                return 0
    return 0

def get_risky_clips(foldernames):
    risky_clips = []
    for folder in foldernames:
        label_file = folder / 'label.txt'
        to_filter = filterClip(folder)
        if not to_filter:
            if label_file.exists():
                with open(str(label_file), 'r') as f:
                    label_data = float(f.read().strip().split(",")[0])
                    if label_data >= 0:
                        risky_clips.append(folder)
    return risky_clips

def get_non_risky_clips(foldernames):
    non_risky_clips = []
    for folder in foldernames:
        label_file = folder / 'label.txt'
        to_filter = filterClip(folder)
        if not to_filter:
            if label_file.exists():
                with open(str(label_file), 'r') as f:
                    label_data = float(f.read().strip().split(",")[0])
                    if label_data < 0:
                        non_risky_clips.append(folder)
    return non_risky_clips

def annotate_task(config):
    root_folder = config.input_base_dir
    starting_folder = config.start
    frame_delay = config.frame_delay
    foldernames = [f for f in root_folder.iterdir() if f.is_dir()]
    foldernames = sorted(foldernames)
    if not config.filter == '':
        print('Showing only {}'.format(config.filter))
        foldernames = [f for f in foldernames if f.name.split('_')[-1] == config.filter]
    if config.risk:
        print('Showing only risky cases')
        foldernames = get_risky_clips(foldernames)
    elif config.nonrisk:
        print('Showing only non-risky cases')
        foldernames = get_non_risky_clips(foldernames)
    
    idx = 0
    print('Clip {} / {}'.format(idx, len(foldernames)))

    def show_video(canvas, clip_folder, root, title, frame_delay):
        # Too slow when dataset is remote
        # get_label_distribution(foldernames)
        im = []
        image_list = list(clip_folder.glob("raw_images/*.jpg"))
        image_list.sort()
        for img in image_list:
            im.append(Image.open(str(img)))
        UI(canvas, im, image_list, root, title, frame_delay).grid(row=0)

    def get_label_distribution(folders):
        risk = 0
        non_risk = 0
        non_annotated = 0
        for folder in folders:
            label_file = folder / 'label.txt'
            to_filter = filterClip(folder)
            if not to_filter:
                if label_file.exists():
                    with open(str(label_file), 'r') as f:
                        label_data = float(f.read().strip().split(",")[0])
                        if label_data >= 0:
                            risk += 1
                        else:
                            non_risk += 1
                else:
                    non_annotated += 1
        print('Total annotated clips={} | Total non-annotated clips={} | Risky clips={} | Non-risky Clips={}'.format(risk+non_risk, non_annotated, risk, non_risk))

    def read_score(path):
        prev_avg_score = 0
        num_of_scores = 0
        if path.exists():
            with open(str(path), 'r') as f:
                label_data = [x for x in f.read().split(',')]
                if len(label_data) == 2:
                    prev_avg_score, num_of_scores = float(label_data[0]), int(label_data[1])
        return prev_avg_score+3, num_of_scores
    
    def clear_canvas():
        for widget in clip_canvas.winfo_children():
            widget.destroy()
        clip_canvas.grid_forget()

    def skipClip(idx, foldernames, op):
        if op == 'nextClip':
            while (idx < len(foldernames)):
                toFilter = filterClip(foldernames[idx])
                if toFilter:
                    idx += 1
                    print('clip has been skipped due to being filtered out')
                else: break;
        if op == 'prevClip':
            while (idx > 0):
                toFilter = filterClip(foldernames[idx])
                if toFilter:
                    idx -= 1
                    print('clip has been skipped due to being filtered out')
                else: break;
        return idx

    def nextClip():
        print("Loading next clip...")
        nonlocal idx
        idx += 1
        idx = skipClip(idx, foldernames, 'nextClip')
        print('Clip {} / {}'.format(idx, len(foldernames)))
        if (idx >= len(foldernames)):
            root.destroy()
            print('Cannot load the next clip (out of bounds)')
            return
        clear_canvas()
        
        prev_avg_score, num_of_scores = read_score(foldernames[idx] / "label.txt")
        
        title = "Lane Change {} Evaluation: Clip {} / {}, curr:({}, {})".format(foldernames[idx].stem, idx, len(foldernames), prev_avg_score, num_of_scores)
        root.title(title)
        show_video(clip_canvas, foldernames[idx], root, title, frame_delay)
    
    def prevClip():
        print("Loading next clip...")
        nonlocal idx
        idx -= 1
        idx = skipClip(idx, foldernames, 'prevClip')
        print('Clip {} / {}'.format(idx, len(foldernames)))
        if (idx >= 0):
            clear_canvas()

            prev_avg_score, num_of_scores = read_score(foldernames[idx] / "label.txt")

            title = "Lane Change {} Evaluation: Clip {} / {}, curr:({}, {})".format(foldernames[idx].stem, idx, len(foldernames), prev_avg_score, num_of_scores)
            root.title(title)
            show_video(clip_canvas, foldernames[idx], root, title, frame_delay)
        else: 
            root.destroy()
            print('Cannot load the previous clip (out of bounds)')
            return

    def replayClip():
        clear_canvas()
        prev_avg_score, num_of_scores = read_score(foldernames[idx] / "label.txt")
        
        title = "Lane Change {} Evaluation: Clip {} / {}, curr:({}, {})".format(foldernames[idx].stem, idx, len(foldernames), prev_avg_score, num_of_scores)
        root.title(title)
        show_video(clip_canvas, foldernames[idx], root, title, frame_delay)

    def jump2Clip():
        jmp_idx = -1
        n = len(foldernames)
        clip_num = int(eval(entry.get()))
        for i in range(n):
            if int(foldernames[i].name.split('_')[0]) == clip_num:
                jmp_idx = i
        nonlocal idx
        if (0 <= jmp_idx < len(foldernames)):
            idx = jmp_idx
            replayClip()
        else:
            print("Error: Invalid index!")

    def saveScore(event=None):  
        x = int(eval(entry.get()))
        if (1 <= x <= 5):
            score = x - 3
            label_file = foldernames[idx] / "label.txt"
            prev_avg_score = 0
            num_of_scores = 0
            if label_file.exists():
                with open(str(label_file), 'r') as f:
                    label_data = [x for x in f.read().split(',')]
                    if len(label_data) == 2:
                        prev_avg_score, num_of_scores = float(label_data[0]), int(label_data[1])
            
            avg_score = ((prev_avg_score * num_of_scores) + score) / (num_of_scores + 1)
            num_of_scores += 1
            
            print("%f stored to %s" %(avg_score, label_file))
            with open(str(label_file), 'w') as f:
                f.write("{}, {}".format(avg_score, num_of_scores))
            
            ignore_file = foldernames[idx] / "ignore.txt"
            if not ignore_file.exists():
                with open(str(ignore_file), 'w') as f:
                    f.write("0")

            entry.delete(0, END)
            nextClip()
        else:
            print("Error: Score not in range.")
    
    def ignoreClip(event=None):  
        ignore_file = foldernames[idx] / "ignore.txt"
        with open(str(ignore_file), 'w') as f:
            f.write("1")
        entry.delete(0, END)
        nextClip()

    root = Tk()
    clip_canvas = Canvas(root, width = 300, height = 200)
    clip_canvas.pack()
    util_canvas = Canvas(root, width = 300, height = 100)
    util_canvas.pack()
    Label(util_canvas, text="Enter a score from 1-5(safe-risk): ").grid(row=0, column=0)
    entry = Entry(util_canvas)
    entry.grid(row=1)
    Button(util_canvas, text='Ignore Clip', command=ignoreClip).grid(row=2, column=3)
    Button(util_canvas, text='Save Score', command=saveScore).grid(row=1, column=1)
    Button(util_canvas, text='Jump to Clip', command=jump2Clip).grid(row=1, column=2)
    Button(util_canvas, text='Replay Clip', command=replayClip).grid(row=2, column=0)
    Button(util_canvas, text='Prev Clip', command=prevClip).grid(row=2, column=1)
    Button(util_canvas, text='Next Clip', command=nextClip).grid(row=2, column=2)
    
    # Display clip stats
    n = len(foldernames)
    print('Originally {} clips'.format(n))
    for f in foldernames:
        f = f /'ignore.txt'
        if f.is_file():
            with open(f) as file:
                data = file.read()
                if int(data):
                    n-=1
    print('{} clips after filter\n'.format(n))
    get_label_distribution(foldernames)

    # enter button saves clip
    root.bind('<Return>', saveScore)
    root.mainloop()

class AppletDisplay:
    def __init__(self, ui):
        self.__ui = ui
    def paste(self, im, bbox):
        self.__ui.image.paste(im, bbox)
    def update(self):
        self.__ui.update_idletasks()

# --------------------------------------------------------------------
# an image animation player

class UI(Label):

    def __init__(self, master, im, image_path_list, root, title, frame_delay):
        self.root = root
        self.title = title
        self.frame_delay = frame_delay

        if type(im) == type([]):
            # list of images
            self.im = im
            im = self.im[0]
            self.image_path_list = image_path_list
            self.index = 0
        else:
            # sequence
            self.im = im

        if im.mode == "1":
            self.image = ImageTk.BitmapImage(self.im[0], foreground="white")
        else:
            self.image = ImageTk.PhotoImage(self.im[0])

        # APPLET SUPPORT (very crude, and not 100% safe)
        global animation_display
        animation_display = AppletDisplay(self)

        Label.__init__(self, master, image=self.image, bg="black", bd=0)
        self.grid(row=0, columnspan=4)
        Button(master, text='Prev Frame', command=self.previousFrame).grid(row=1, column=0)
        Button(master, text='Pause', command=self.pause).grid(row=1, column=1)
        Button(master, text='Resume', command=self.resume).grid(row=1, column=2)
        Button(master, text='Next Frame', command=self.nextFrame).grid(row=1, column=3)
        Button(master, text="Del Before", command=self.deleteBefore).grid(row=2, column=1)
        Button(master, text="Del After", command=self.deleteAfter).grid(row=2, column=2)
       
        self.update()

        try:
            duration = im.info["duration"]
        except KeyError:
            duration = 100
        self.paused = False
        self.after(1, self.next)
    
    def update_title(self):
        new_title = self.title + ", Frame: {} / {}".format(self.index, len(self.image_path_list) - 1)
        self.root.title(new_title)
        pass

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        try:
            im = self.im[self.index]
            duration = im.info["duration"]
        except KeyError:
            duration = 100
        self.after(1, self.next)

    def previousFrame(self):
        if self.paused and self.index > 0:
            self.index -= 1
            im = self.im[self.index]
            self.image.paste(im)
            self.update_title()
    
    def nextFrame(self):
        if self.paused and (self.index < len(self.image_path_list) - 1):
            self.index += 1
            im = self.im[self.index]
            self.image.paste(im)
            self.update_title()
    
    def deleteBefore(self):
        # delete every images before current index
        if self.paused:
            delete_path_list = self.image_path_list[:self.index]

            for _ in range(self.index):
                self.im[0].close()
                del self.im[0]

            for img_path in delete_path_list:
                img_path.unlink()
            
            self.image_path_list = self.image_path_list[self.index:]
            self.index = 0
            self.update_title()

    def deleteAfter(self):
        # delete every images after current index
        if self.paused:
            delete_path_list = self.image_path_list[self.index + 1:]

            for _ in range(len(self.image_path_list) - self.index - 1):
                self.im[-1].close()
                del self.im[-1]

            for img_path in delete_path_list:
                img_path.unlink()
            
            self.image_path_list = self.image_path_list[:self.index + 1]
            self.index = 0
            self.update_title()

    def next(self):
        if not self.paused:
            if type(self.im) == type([]):

                try:
                    if self.index < len(self.image_path_list) - 1:
                        self.index += 1
                        im = self.im[self.index]
                        self.image.paste(im)
                    else:
                        self.paused = True
                        im = self.im[-1]
                except IndexError:
                    return # end of list

            else:

                try:
                    im = self.im
                    im.seek(im.tell() + 1)
                    self.image.paste(im)
                except EOFError:
                    return # end of file

            try:
                duration = im.info["duration"]
            except KeyError:
                duration = 100
            self.update_title()
            self.after(self.frame_delay, self.next)

        self.update_idletasks()


if __name__ == "__main__":
    config = Config(sys.argv[1:])
    annotate_task(config)
	

