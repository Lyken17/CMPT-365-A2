/**
 * Created by Syu on 2/22/16.
 */
function changeBody(index) {
    switch (index) {
        case 1:
        {
            document.getElementById('myclass1').style.display = "";
            document.getElementById('myclass2').style.display = "none";
            document.getElementById('myclass3').style.display = "none";
            document.getElementById('myclass4').style.display = "none";
            break;
        }
        case 2:
        {
            document.getElementById('myclass1').style.display = "none";
            document.getElementById('myclass2').style.display = "";
            document.getElementById('myclass3').style.display = "none";
            document.getElementById('myclass4').style.display = "none";
            break;
        }
        case 3:
        {
            document.getElementById('myclass1').style.display = "none";
            document.getElementById('myclass2').style.display = "none";
            document.getElementById('myclass3').style.display = "";
            document.getElementById('myclass4').style.display = "none";
            break;
        }
        case 4 :
        {
            document.getElementById('myclass1').style.display = "none";
            document.getElementById('myclass2').style.display = "none";
            document.getElementById('myclass3').style.display = "none";
            document.getElementById('myclass4').style.display = "";
            break;
        }
    }
}
