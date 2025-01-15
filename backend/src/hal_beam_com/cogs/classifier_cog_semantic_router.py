from semantic_router.encoders import HuggingFaceEncoder
from semantic_router import Route
from semantic_router.layer import RouteLayer
import time
from . import classifier_cog

from src.hal_beam_com.utils import cog_output_fields, set_data_fields

# cog_output_fields = {

#     0:'voice_cog_output',
#     1:'text_output',
#     2: 'next_cog',
#     3: 'op_cog_output',
#     4: 'classifier_cog_output'
# }

encoder = HuggingFaceEncoder()

operation = Route(
    name="Op",
    utterances=[
        "Measure for 1 second at theta 0.12.",
        "Sample alignment.",
        "Align the sample.",
        "Measure sample for 5 seconds but don't save the data.",
        "Measure sample for 5 seconds.",
        "Move sample x to 1.5.",
        "Set incident angle to 0.2 degrees.",
        "Move sample x by 1.5.",
        "Increase incident angle by 0.1 degrees.",
        "Set heating ramp to 2 degrees per minute.",
        "Set temperature to 50 degrees Celsius.",
        "What is the sample temperature?",
        "What are the sample motor positions?",
        "Increase the temperature to 250 degrees at a ramp rate of 2 degrees per minute.",
        "Go to 300 degrees directly.",
        "New sample is ps-pmma",
        "Sample is perovskite."
        "Sample is BCP."
    ],
)

# this could be used as an indicator to our chatbot to switch to a more
# conversational prompt
analysis = Route(
    name="Ana",
    utterances=[
        "I want to look at the data, how does the measurement look?",
        "I want to see the qr image.",
        "Show me the q image.",
        "I want to see the circular average, where is the peak?",
        "Show me linecut qr at qz=0.1, thickness=2.",
        "Show me linecut qz at qr=1, thickness=1.",
        "I want to see the linecut angle at q=0.1.",
        "Show me the sector average in the upper right corner.",
        "Where is the peak for circular average?",
        "Analyze the diffraction patterns and perform peak fitting."
    ],
)

notebook = Route(
    name="notebook",
    utterances=[
        "Record that we observed unexpected peaks at high temperatures.",
        "Make a note that the sample color changed.",
        "Saving information",
        "Document procedure",
    ],
)

gpcam = Route(
    name="gpcam",
    utterances=[
        "We need gpcam",
        "Start gp cam",
        "Can we start Tsuchinoko",
        "Start gpCAM",
        "Let's use gp cam",
        "We want to use gaussian process",
        "We need Tsuchinoko to check autonomous experiment progress",
    ],
)

xicam = Route(
    name="xicam",
    utterances=[
        "Start xicam",
        "We need xi cam",
        "Check data with xi-cam",
    ],
)

routes = [operation, analysis, notebook, gpcam, xicam]

rl = RouteLayer(encoder=encoder, routes=routes)


def semantic_router(data):
    if data[0]['only_text_input'] == 1:
        user_prompt = data[0]['text_input']
        print(user_prompt)

    else:
        last_cog_id = data[0]['last_cog_id']
        cog_output_field = cog_output_fields[last_cog_id]

        user_prompt = data[0]['text_input'] + data[0][cog_output_field]

        print("THE USER PROMPT IS {}".format(user_prompt))

    return rl(user_prompt).name, user_prompt


def invoke(data, base_model='mistral', finetuned=False):
    start_time = time.time()
    next_cog, user_prompt = semantic_router(data)
    end_time = time.time()
    execution_time = end_time - start_time

    if next_cog is None:
        print("Semantic router missed this!")
        data = classifier_cog.invoke(data, base_model=base_model, finetuned=False)

    else:
        data[0]['next_cog'] = next_cog

        data[0]['classifier_cog_output'] = next_cog

        data[0]['classifier_cog_time'] = f"{execution_time:.5f} seconds"

        print(data[0]['classifier_cog_output'])

        data[0]['last_cog_id'] = 4

        rdb_data = {

            'model': 'Semantic Router - Hugging Face Encoder',
            'input': user_prompt,
            'output': next_cog,
            'start_time': f"{start_time:.5f} seconds",
            'end_time': f"{end_time:.5f} seconds",
            'execution_time': f"{execution_time:.5f} seconds"
        }

        data = set_data_fields('classifier', rdb_data, data)

    return data
