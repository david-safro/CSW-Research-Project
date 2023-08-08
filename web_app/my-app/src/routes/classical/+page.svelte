<script>
    import DataInput from '$lib/components/DataInput.svelte';
    import Typewriter from '$lib/components/Typewriter.svelte';


    const sexOptions = ['Male', 'Female', "Other"];
    const painOpt = [1,2,3,4]
    const EKG = [0,1,2]
    const ANGINA = ["Yes", "No"]

    // Repeat the same pattern for other fields...

    let formData = {};

    function handleSubmit() {
        console.log(formData);
    }
</script>

<style>
    @import '/dark-theme.css';
</style>

<main class="centered">
    <div class="title">
        <Typewriter text="Heart Disease Prediction" fontSize="3rem" />
    </div>

    <form on:submit|preventDefault={handleSubmit} class="data-input-form">
        <DataInput title="Age" description="The age of the patient." inputType="number"/>
        <DataInput title="Sex" description="The gender of the patient." inputType="dropdown" options={sexOptions} bind:data={formData.sex} />
        <DataInput title="Chest Pain" description="The type of chest pain experienced by the patient" inputType="dropdown" options={painOpt} bind:data={formData.pain} />
        <DataInput title="Blood Pressure" description="The blood pressure level of the patient." inputType="number"/>
        <DataInput title="Cholesterol" description="The cholesterol level of the patient." inputType="number"/>
        <DataInput title="FBS over 120" description="The fasting blood sugar test results over 120 mg/dl. (Numeric)" inputType="number"/>
        <DataInput title="EKG Results" description="The electrocardiogram results of the patient." inputType="dropdown" options={EKG} bind:data={formData.ekg} />
        <DataInput title="Max HR" description="The maximum heart rate levels achieved during exercise testing." inputType="number"/>
        <DataInput title="Exercise Angina" description="The angina experienced during exercise testing." inputType="dropdown" options={ANGINA} bind:data={formData.angina} />
        <span>
        <button type="submit" class="submit-button">
      Submit
    </button>
    </form>
</main>
