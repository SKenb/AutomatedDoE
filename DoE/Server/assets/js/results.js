function getSelectedExperiment() {
    selectElement = document.getElementById('selectedExperiment');

    if(selectElement.selectedIndex <= 0) return null;
    return selectElement.options[selectElement.selectedIndex].text 
}

function removeExperiment() {
    selectedExperiment = getSelectedExperiment();

    if(selectedExperiment) {
        if(confirm('Are you sure you want to remove (' + selectedExperiment + ')')) {
            sendAction('remove/' + selectedExperiment)
            window.location.reload(); 
        }
    }   
    else {
        alert("Nothing selected :D");
    }      
}


function selectedExperimentChanged() {

    updateDataFromObject({'selectedExperiment': getSelectedExperiment() }, document);

    a = Array.from(document.getElementById('plots').children);
    a = a.filter(e => !e.classList.contains('template'));
    a.forEach(e => e.remove());

    updateDataFrom('plots.json/' + getSelectedExperiment());
}

function plotChanged() {
    selectElement = document.getElementById('plots');

    //if(selectElement.selectedIndex <= 0) return null;

    var path = selectElement.options[selectElement.selectedIndex].getAttribute('data-info'); 
    
    console.log(path);
    document.getElementById('plot').src = '../' + path;
}