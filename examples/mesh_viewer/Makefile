FRAMES         := /tmp/frames
MAX_FRAMES     := 100
MESH           := $(CURDIR)/data/template-centered.obj
MOVEMENT_SPEED := 1.0

all: video

clean:
	$(RM) --recursive $(FRAMES)
	$(RM) $(CURDIR)/output.mp4

view: $(CURDIR)/output.mp4
	vlc $^

video: $(CURDIR)/output.mp4

.PHONY: frames
frames: $(CURDIR)/main.py $(MESH)
	python $< --max-frames $(MAX_FRAMES) --movement-speed $(MOVEMENT_SPEED) --off-screen --output-prefix $(FRAMES) $(MESH)

$(CURDIR)/output.mp4: frames
	ffmpeg -framerate 30 -pattern_type glob -i '$(FRAMES)/*.png' -c:v libx264 -pix_fmt yuv420p $@
