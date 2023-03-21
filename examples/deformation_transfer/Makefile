DATA    := $(CURDIR)/data
TI_ARCH := x64 # cuda, vulkan, ...

# Adaptive Moment Estimation (Adam) params
BETA_1  := 0.9
BETA_2  := 0.999
EPSILON := 1e-8
ETA     := 1e-3
ITERS   := 131072

PYTHON           := env TI_ARCH=$(TI_ARCH) python
GRADIENT_DESCENT := $(PYTHON) gradient_descent.py --beta-1 $(BETA_1) --beta-2 $(BETA_2) --epsilon $(EPSILON) --eta $(ETA) --iters $(ITERS)

all: t1 u1

t1: $(DATA)/t1.obj

u1: $(DATA)/u1.obj

view: $(DATA)/s0.obj $(DATA)/s1.obj $(DATA)/t0.obj $(DATA)/t1.obj $(DATA)/u0.obj $(DATA)/u1.obj
	meshlab $^

clean:
	$(RM) $(DATA)/t1-*.obj
	$(RM) $(DATA)/t1.obj
	$(RM) $(DATA)/u0-aligned-*.obj
	$(RM) $(DATA)/u0-aligned.obj
	$(RM) $(DATA)/su-transform.txt
	$(RM) $(DATA)/u1-aligned-*.obj
	$(RM) $(DATA)/u1-aligned.obj
	$(RM) $(DATA)/u1.obj

$(DATA)/t1.obj: $(DATA)/s0.obj $(DATA)/t0.obj $(DATA)/s1.obj
	$(GRADIENT_DESCENT) \
	  --source-reference $(DATA)/s0.obj \
	  --target-reference $(DATA)/t0.obj \
	  --input $(DATA)/s1.obj \
	  --output $@

$(DATA)/su-transform.txt: $(DATA)/s0.obj $(DATA)/u0.obj
	$(PYTHON) align.py \
	  --source $(DATA)/s0.obj \
	  --target $(DATA)/u0.obj \
	  --output $@ \
	  --iters 4096

$(DATA)/u0-aligned.obj: $(DATA)/u0.obj $(DATA)/su-transform.txt
	$(PYTHON) transform.py \
	  --input $(DATA)/u0.obj \
	  --output $@ \
	  --transform $(DATA)/su-transform.txt \
	  --inverse

$(DATA)/u1-aligned.obj: $(DATA)/s0.obj $(DATA)/u0-aligned.obj $(DATA)/s1.obj
	$(GRADIENT_DESCENT) \
	  --source-reference $(DATA)/s0.obj \
	  --target-reference $(DATA)/u0-aligned.obj \
	  --input $(DATA)/s1.obj \
	  --output $@

$(DATA)/u1.obj: $(DATA)/u1-aligned.obj $(DATA)/su-transform.txt
	$(PYTHON) transform.py \
	  --input $(DATA)/u1-aligned.obj \
	  --output $@ \
	  --transform $(DATA)/su-transform.txt
