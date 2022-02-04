function switchPage(subPage) {
    document.getElementById('page').src = "./assets/" + subPage + ".html";
}

function tryThis(predicate, onError=null) {
    try {
        predicate()
    } catch (error) {
        if(onError) {
            onError(error)
        }
        else {
            console.error(error)
        }
    }
}


function updateDataFrom(endpoint) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            updateData(xmlHttp.responseText);
    }
    xmlHttp.open("GET", endpoint, true);
    xmlHttp.send(null);
}

function updateData(jsonData) {
    tryThis(
        () => {
            infos = JSON.parse(jsonData)

            for (const [key, value] of Object.entries(infos)) {
                
                updateElements(document, key, value);

                //iframe = document.getElementsByTagName("iframe")[0];
                //iframeContent = iframe.contentDocument || iframe.contentWindow.document;
                //updateElements(iframeContent, key, value);
            }
        }
    )
}

function updateElements(parent, identifier, value) {
    elements = parent.getElementsByClassName(identifier);
    [].forEach.call(elements, el => { 
        el.innerHTML = value; 

        if(el.classList.contains("variable")) {
            el.classList.add("visible")
        }
    }); 
}

