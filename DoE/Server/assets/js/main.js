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

function updateData(jsonData, dom=null) {
    if(!dom) dom = document;

    tryThis(
        () => {
            updateDataFromObject(JSON.parse(jsonData), dom);
        }
    )
}

function updateDataFromObject(obj, dom) {
    for (const [key, value] of Object.entries(obj)) {

        if(Array.isArray(value)) {
            updateArrayElements(dom, key, value)
        }
        else {
            updateElements(dom, key, value);
        }
    }
}

function updateArrayElements(parent, identifier, array) {
    elements = parent.getElementsByClassName(identifier);
    [].forEach.call(elements, container => { 

        template = container.getElementsByClassName('template')[0];
    
        array.forEach(infos => {
            clone = template.cloneNode(true);
            clone.classList.remove("template");
            container.appendChild(clone);
    
            console.log(infos)
            updateDataFromObject(infos, clone);
        });

    });

   
}

function updateElements(parent, identifier, value) {

    elements = parent.getElementsByClassName(identifier);
    [].forEach.call(elements, el => { 
        
        if(el instanceof HTMLInputElement) {
            el.setAttribute('value', value);
        }
        else {
            el.innerHTML = value; 
        }

        variableHandling(el);
        console.log(el.parentElement)
        if(el.parentElement) variableHandling(el.parentElement);
    }); 
}

function variableHandling(element) {
    if(element.classList.contains("variable")) {
        element.classList.add("visible");
    }
}

function update(endpoint, dict, callback) {

    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            if(callback) callback(xmlHttp.responseText);
    }
    xmlHttp.open("GET", "./update/" + endpoint + "?" + serialize(dict), true);
    xmlHttp.send(null);
}

function serialize(obj) {
    var str = [];
    for(var p in obj)
        if (obj.hasOwnProperty(p)) {
            str.push(encodeURIComponent(p) + "=" + encodeURIComponent(obj[p]));
        }
    return str.join("&");
}

function updatePaths() {

    writePath = document.getElementsByClassName('writeXamControl')[0].value;
    readPath = document.getElementsByClassName('readXamControl')[0].value;

    dict = {
        "write": writePath,
        "read": readPath,
    }

    update("paths", dict)
}