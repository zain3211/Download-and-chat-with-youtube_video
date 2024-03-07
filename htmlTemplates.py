css = '''
<style>
body {
    background-color: black;
}
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color:#475063
}
.chat-message.bot {
    background-color: #778899
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 65px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.internetandtechnologylaw.com/files/2019/06/iStock-872962368-chat-bots.jpg" >
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://as1.ftcdn.net/v2/jpg/01/28/27/56/1000_F_128275667_WMytNP1mqhkZKNXYwDLgEiY3WAotaGHw.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
