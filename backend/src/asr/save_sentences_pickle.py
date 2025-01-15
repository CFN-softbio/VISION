import pickle
import os

from src.asr.audio_utils import (
    
    populate_sentences,
    domain_specific_pronounciations, 
    text_to_speech,
    load_pickle,
    get_phonetics
)

def save_sentences_pickle(word, base_dir, testing = False):

    if testing != True:
        word_sentences = [
        "Initialize {word} for this experiment.",
        "Run {word} analysis.",
        "Activate {word} for autonomous control.",
        "Engage {word} for the current session.",
        "Launch {word} for data modeling.",
        "Prepare {word} for sample testing.",
        "Use {word} to assess the new results.",
        "Set up {word} for optimization tasks.",
        "Start the autonomous process using {word}.",
        "Deploy {word} to monitor progress.",
        "Configure {word} for advanced analysis.",
        "Use {word} for predictive modeling.",
        "Train the system using {word} techniques.",
        "Run tests on the samples using {word}.",
        "Analyze data with {word} for better insights.",
        "Launch autonomous mode with {word} enabled.",
        "Employ {word} for error diagnostics.",
        "Prepare the experiment protocol with {word}.",
        "Use {word} to optimize the workflow.",
        "Analyze the data set using {word}.",
        "Start the {word} module for computations.",
        "Set {word} to execute its default routine.",
        "Use {word} to calibrate the system.",
        "Initialize {word} for high-speed data processing.",
        "Start {word} for the performance tests.",
        "Monitor the process using {word} algorithms.",
        "Use {word} to simulate the test environment.",
        "Start {word} for generating sample predictions.",
        "Prepare the {word} tool for real-time analysis.",
        "Deploy {word} to validate experimental data.",
        "Activate {word} to process incoming data while {word} simultaneously predicts trends for real-time adjustments.",
        "Use {word} to validate experimental results, and let {word} generate detailed visualizations to share findings with collaborators.",
        "Prepare {word} to handle real-time simulations where {word} adjusts parameters dynamically based on feedback.",
        "Employ {word} as the primary tool for monitoring progress while leveraging {word}’s predictive analytics for error correction.",
        "Deploy {word} for a multi-stage process, and analyze the results through {word} to identify potential improvements.",
        "Use {word} for high-dimensional data analysis and allow {word} to recommend the next steps in the experimental workflow.",
        "Set up {word} to integrate with legacy systems, ensuring {word} can process both historical and live data streams.",
        "Run {word} for sample preparation tasks while simultaneously using {word} to identify anomalies in data trends.",
        "Train the system with {word} algorithms while enabling {word} to optimize future performance through predictive modeling.",
        "Engage {word} for both real-time monitoring and post-experiment analysis to utilize {word}’s full capabilities.",
        "Prepare {word} to predict errors during live runs and allow {word} to make corrections autonomously without manual intervention.",
        "Start {word} to process large-scale datasets, and use {word} to generate comprehensive reports summarizing all findings.",
        "Activate {word} for parameter tuning, and let {word} validate the results by cross-referencing multiple datasets.",
        "Leverage {word} for initial system diagnostics and depend on {word} to suggest optimizations for smoother operations.",
        "Configure {word} to execute predictive analytics, then use {word} to compare outcomes with expected benchmarks.",
        "Deploy {word} for real-time simulations while {word} simultaneously identifies inefficiencies in experimental workflows.",
        "Employ {word} to align experimental objectives with outcomes, ensuring {word}’s models drive consistent improvements.",
        "Use {word} to preprocess the data and then employ {word} to analyze trends for actionable insights."
    ]

    else:
        word_sentences = [
        "Set {word} in motion for this analysis.",
        "Execute the {word} workflow.",
        "Turn on {word} to manage autonomous functions.",
        "Harness {word} for the ongoing session.",
        "Initiate {word} for building data models.",
        "Ready {word} for testing the samples.",
        "Utilize {word} to interpret recent data.",
        "Configure {word} for task-specific optimization.",
        "Begin the automated routine with {word}.",
        "Deploy {word} to track the experiment's status.",
        "Optimize settings for advanced tasks using {word}.",
        "Leverage {word} for forecasting and modeling.",
        "Apply {word} techniques to train the system.",
        "Test samples through {word} procedures.",
        "Process data through {word} to uncover insights.",
        "Enable automated functions with {word}.",
        "Use {word} to diagnose system inefficiencies.",
        "Draft the protocol by integrating {word}.",
        "Streamline workflows using {word} tools.",
        "Examine datasets comprehensively with {word}.",
        "Trigger the {word} module for detailed processing.",
        "Assign {word} its core operations.",
        "Calibrate the device using {word} methodologies.",
        "Kick off high-performance computations with {word}.",
        "Conduct performance benchmarking via {word}.",
        "Track progress in the pipeline using {word} functions.",
        "Simulate environmental tests with {word} solutions.",
        "Activate {word} to generate outcome predictions.",
        "Prepare the system with {word} for on-the-fly analysis.",
        "Implement {word} for validating experimental workflows."
    ]


    populated_sentences = populate_sentences(word, word_sentences)

    print(populated_sentences[0])

    word_folder = f"{base_dir}{word}"
    os.makedirs(word_folder, exist_ok=True)
    pickle_file = os.path.join(word_folder, "sentences.pkl")

    # Save the list of sentences to the pickle file
    with open(pickle_file, "wb") as file:
        pickle.dump(populated_sentences, file)

    print(f"List of sentences successfully saved to {pickle_file}")

    return pickle_file