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

        array.forEach(infos => {
            clonedObj = clone(container);
            updateDataFromObject(infos, clonedObj);
        });

    });

   
}

function cloneFactorTemplate() {
    clonedObj = clone(document.getElementsByClassName('factors')[0]);
    varElements = clonedObj.getElementsByClassName("variable");

    [].forEach.call(varElements, vE => {
        vE.classList.add("visible");
    })
}

function clone(container) {
    template = container.getElementsByClassName('template')[0];

    clonedObj = template.cloneNode(true);
    clonedObj.classList.remove("template");
    container.appendChild(clonedObj);

    return clonedObj
}

function updateElements(parent, identifier, value) {

    elements = Array.from(parent.getElementsByClassName(identifier));

    if(typeof parent.classList !== "undefined" && parent.classList.contains(identifier)) 
        elements.push(parent);

    elements.forEach(el => { 
        
        if(el.classList.contains('changeVisibility')) {
            boolValue = Boolean(value);
            if(el.classList.contains('inverse')) boolValue = !boolValue;

            el.style.visibility = boolValue ? 'visible' : 'collapse';
        }        
        else if(el.classList.contains('changeDisplay')) {
            boolValue = Boolean(value);
            if(el.classList.contains('inverse')) boolValue = !boolValue;

            el.style.display = boolValue ? 'inherit' : 'none';
        }
        else if(el instanceof HTMLInputElement) {
            el.setAttribute('value', value);
        }
        else if(el instanceof HTMLOptionElement && el.getAttribute('data-info')?.includes(identifier)) {
            el.setAttribute('data-info', value);
        }
        else {
            el.innerHTML = value; 
        }

        variableHandling(el);
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

function updateFactors() {

    childs = Array.from(document.getElementsByClassName('factors')[0].children);
    childs.shift()

    dict = {
        "factors":
        childs.map(c => { 
            return {
                "name": c.children[0].children[0].value,
                "symbol": c.children[1].children[0].value,
                "unit": c.children[2].children[0].value,
                "min": c.children[3].children[0].value,
                "max": c.children[4].children[0].value,
            }
        })
    }

    var xmlhttp = new XMLHttpRequest();
    xmlhttp.open("POST", "/update/factors");
    xmlhttp.setRequestHeader("Content-Type", "application/json");
    xmlhttp.send(JSON.stringify(dict));
}

function sendAction(action, callback) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            if(callback) callback(xmlHttp.responseText);
    }
    xmlHttp.open("GET", "/action/" + action, true);
    xmlHttp.send(null);
}

function exportCallback(result) {
    tryThis(() => {
        obj = JSON.parse(result);
        if(obj.exportPath) {
            window.open("../" + obj.exportPath.replace("\\", "/"), '_blank').focus();
        }
    });
}

function loadFileContent(hanlder=null) {
    var file = document.getElementById("fileForImport")?.files[0];

    if (file) {
        var reader = new FileReader();
        reader.readAsText(file, "UTF-8");
        reader.onload = (evt) => { if(hanlder) hanlder(evt.target.result) };
        reader.onerror = (evt) => alert("Failed import file :/");    
    }
    else {
        alert("No file 0.o");
    }
}

function contentImport(content) {
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.open("POST", "/import/data");
    xmlhttp.setRequestHeader("Content-Type", "application/json");
    xmlhttp.send(JSON.stringify(content));
}
