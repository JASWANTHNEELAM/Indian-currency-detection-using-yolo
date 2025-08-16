let englishtext = "";
let hinditext = "";
let notesData = {};
let confidenceData = {};

window.onload = () => {
	$('#sendbutton').click(() => {
		englishtext = "";
		hinditext = "";
		notesData = {};
		confidenceData = {};
		hideInterface();

		imagebox = $('#imagebox')
		input = $('#imageinput')[0]
		if(input.files && input.files[0])
		{
			let formData = new FormData();
			formData.append('image' , input.files[0]);
			$.ajax({
				url: "http://127.0.0.1:8080/detectObject", 
				// fix below to your liking
				// url: "http://xxx.xxx.xxx.xxx:8080/detectObject", 
				type:"POST",
				data: formData,
				cache: false,
				processData:false,
				contentType:false,
				error: function(data){
					console.log("upload error" , data);
					console.log(data.getAllResponseHeaders());

					updateInterface();
				},
				success: function(data){
					console.log(data);
					bytestring = data['status'];
					image = bytestring.split('\'')[1];
					englishtext = data['englishmessage'];
					hinditext = data['hindimessage'];
					notesData = data['raw_labels'];
					confidenceData = data['confidence_scores'];
					
					imagebox.attr('src' , 'data:image/jpeg;base64,'+image);
					
					// Display the text on the interface
					$('#english-text').text(englishtext);
					$('#hindi-text').text(hinditext);
					
					// Display the detected notes with confidence scores
					displayDetectedNotes(notesData, confidenceData);
					
					// Calculate and display total amount
					displayTotalAmount(notesData);
					
					updateInterface();					
				}
			});
			
		}
	});
};

function displayDetectedNotes(notesData, confidenceData) {
	const notesDisplay = $('#notes-display');
	notesDisplay.empty();
	
	if (!notesData || Object.keys(notesData).length === 0) {
		notesDisplay.html('<p class="no-notes">No currency notes detected</p>');
		return;
	}
	
	let notesGridHTML = '<div class="notes-grid">';
	
	// Create cards for each type of note
	for (const [noteType, count] of Object.entries(notesData)) {
		// Extract the rupee value from the note type (e.g., "10Rupees" -> "10")
		const rupeeValue = noteType.replace("Rupees", "");
		
		// Get confidence score if available
		const confidence = confidenceData && confidenceData[noteType] ? 
			confidenceData[noteType] : 0;
		
		const noteCardHTML = `
			<div class="note-card">
				<div class="note-value">₹${rupeeValue}</div>
				<div class="note-count">${count} ${count > 1 ? 'notes' : 'note'}</div>
				<div class="note-confidence">Confidence: ${confidence}%</div>
			</div>
		`;
		
		notesGridHTML += noteCardHTML;
	}
	
	notesGridHTML += '</div>';
	notesDisplay.html(notesGridHTML);
}

function displayTotalAmount(notesData) {
    const totalDisplay = $('#total-amount');
    
    if (!notesData || Object.keys(notesData).length === 0) {
        totalDisplay.html('');
        return;
    }
    
    let totalAmount = 0;
    
    // Calculate total value
    for (const [noteType, count] of Object.entries(notesData)) {
        const rupeeValue = parseInt(noteType.replace("Rupees", ""));
        totalAmount += rupeeValue * count;
    }
    
    const totalHTML = `
        <div class="total-amount">
            <h3>Total Amount:</h3>
            <div class="amount-value">₹${totalAmount}</div>
        </div>
    `;
    
    totalDisplay.html(totalHTML);
}

function readUrl(input){
	imagebox = $('#imagebox');
	console.log("evoked readUrl");
	if(input.files && input.files[0]){
		let reader = new FileReader();
		reader.onload = function(e){
			imagebox.attr('src',e.target.result); 
			//change image dimensions
			resizeImage();
		}
		reader.readAsDataURL(input.files[0]);
	}
}

function myResizeFunction2(y){
	if (y.matches) {
		imagebox.width(640);
		imagebox.height(640);
	}
	else {
		imagebox.width(940);
		imagebox.height(740);
	}
}
function myResizeFunction1(x) {
	imagebox = $('#imagebox');
	
	if(x.matches){
		imagebox.width(360);
		imagebox.height(360);
	}
	else{ 
		let y = window.matchMedia("(max-width:1050px)");
		myResizeFunction2(y);
		y.addListener(myResizeFunction2);//attach event listener on every change
	}
}
function resizeImage(){
	
	let x = window.matchMedia("(max-width:700px)");
	myResizeFunction1(x);
	x.addListener(myResizeFunction1);
	
}

function hideInterface(){
	$(".loading").hide();
	let progresstext = document.querySelector('.text');
	progresstext.style.display = "none";//hide completed text
	$('#result-section').hide();
}

function updateInterface(){
	$(".loading").show();
	progress();

	// show result section after processing
	setTimeout(
		function () {
			showResults();
		}, 3000
	);

	function showResults() {
		$('#result-section').show();
	}
}

function progress() {
	let percent = document.querySelector('.percent');
	let progress = document.querySelector('.progress');
	let text = document.querySelector('.text');
	let count = 12;
	let per = 8;
	let loading = setInterval(animateProgress, 50);

	function animateProgress() {
		if (count == 100 && per == 360) {
			percent.classList.add("text-blink")
			percent.innerText = "Processing Completed!"
			percent.style.fontSize = "20px";

			text.style.display = "block";
			clearInterval(loading);
		}
		else {
			per = per + 4;
			count = count + 1;
			progress.style.width = per + 'px';
			percent.innerText = count + '%';
		}
	}
}

function changeColor(){
	let sendButton = document.querySelector("#sendbutton");
	sendButton.style.backgroundColor = "orange";
	sendButton.style.color = "black";
}