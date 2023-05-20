import streamlit as st

def main():
    st.set_page_config(page_title="Picture Description App", layout="wide")
    
    st.sidebar.title("Tabs")
    tabs = ["Picture", "About"]
    selected_tab = st.sidebar.selectbox("Select a tab", tabs)
    
    if selected_tab == "Picture":
        st.title("Picture")
        st.write("Here is the picture:")
        
        picture_url = 'drilling_rig.JPG'  # Replace with the URL of your picture
        st.image(picture_url, use_column_width=True)
        
        st.write("Click on the names to view the description of each part.")
        
        # Define the parts of the picture and their descriptions
        parts = {
            "Part A": "Description of Part A.",
            "Part B": "Description of Part B.",
            "Part C": "Description of Part C.",
        }
        
        # Display the part names and descriptions
        for part_name, description in parts.items():
            col1, col2 = st.columns([0.4, 0.6])
            
            with col1:
                st.write(part_name)
            
            with col2:
                # Get the description of the part when its name is clicked
                if st.button(part_name):
                    st.write(description)
    
    else:
        st.title("About")
        st.write("This app was created by [Your Name] as a Streamlit exercise.")
    
if __name__ == "__main__":
    main()
