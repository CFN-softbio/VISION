from tests.utils import print_comparison


prediction = """for temp in np.arange(25, 250+10, 10):
    sam.setTemperature(temp)
    sam.measure(1)"""
reference = prediction

print_comparison(reference, prediction)


# Unsure why these do not evaluate to 1.
prediction = "a=1\nb=1\nprint(a+b)"
reference = prediction

print_comparison(reference, prediction)

# Example of one that gives a root warning
prediction = "print(a+b)"
reference = prediction

print_comparison(reference, prediction)


# Previous test that TSED failed on:
prediction = "sam.xr(-0.5); sam.align()"
reference = "sam.xr(-0.5)"

print_comparison(reference, prediction)

# Sanity check
prediction = "sam.xr(-0.5); sam.align()"
reference = prediction

print_comparison(reference, prediction)

prediction = "sam.xr(-0.5)\nsam.align()"
reference = prediction

print_comparison(reference, prediction)

prediction = "sam.xr(-0.5)\nsam.align()"
reference = "sam.xr(-0.5); sam.align()"

print_comparison(reference, prediction)

prediction = "sam.xr(-0.5)\nsam.align()"
reference = "sam.xr(-0.5); sam.misalign()"

print_comparison(reference, prediction)

prediction = "sam.xr(-0.5)\nsam.align()"
reference = "sam.xr(-0.5)\nsam.misalign()"

print_comparison(reference, prediction)
