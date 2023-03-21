DATA := $(CURDIR)/data

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
	python -m taichi_extras.examples.deformation_transfer.gradient_descent \
	  --source-reference $(DATA)/s0.obj \
	  --target-reference $(DATA)/t0.obj \
	  --input $(DATA)/s1.obj \
	  --output $@ \
	  --iters 65536

$(DATA)/su-transform.txt: $(DATA)/s0.obj $(DATA)/u0.obj
	python -m taichi_extras.examples.deformation_transfer.align \
	  --source $(DATA)/s0.obj \
	  --target $(DATA)/u0.obj \
	  --output $@ \
	  --iters 4096

$(DATA)/u0-aligned.obj: $(DATA)/u0.obj $(DATA)/su-transform.txt
	python -m taichi_extras.examples.deformation_transfer.transform \
	  --input $(DATA)/u0.obj \
	  --output $@ \
	  --transform $(DATA)/su-transform.txt \
	  --inverse

$(DATA)/u1-aligned.obj: $(DATA)/s0.obj $(DATA)/s1.obj $(DATA)/u0-aligned.obj
	python -m taichi_extras.examples.deformation_transfer.gradient_descent \
	  --source-reference $(DATA)/s0.obj \
	  --target-reference $(DATA)/u0-aligned.obj \
	  --input $(DATA)/s1.obj \
	  --output $@ \
	  --iters 65536

$(DATA)/u1.obj: $(DATA)/u1-aligned.obj $(DATA)/su-transform.txt
	python -m taichi_extras.examples.deformation_transfer.transform \
	  --input $(DATA)/u1-aligned.obj \
	  --output $@ \
	  --transform $(DATA)/su-transform.txt
