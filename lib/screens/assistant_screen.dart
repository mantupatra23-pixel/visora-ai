import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../services/api_service.dart';

class AssistantScreen extends StatefulWidget {
  const AssistantScreen({super.key});
  @override
  State<AssistantScreen> createState() => _AssistantScreenState();
}

class _AssistantScreenState extends State<AssistantScreen> {
  final TextEditingController _ctrl = TextEditingController();
  final List<Map<String,String>> _messages = [{'role':'assistant','text':'Namaste! Kaise madad karu?'}];
  bool _loading = false;
  String _tone = 'Motivational';

  Future<void> _sendPrompt() async {
    final txt = _ctrl.text.trim();
    if (txt.isEmpty) return;
    setState(()=>_loading=true);
    _messages.add({'role':'user','text':txt});
    _ctrl.clear();
    // placeholder: call backend or local generator
    try {
      // example: ApiService.get('/api/ai/generate?prompt=...')
      await Future.delayed(const Duration(seconds:1));
      final resp = 'Generated script (tone: $_tone) for: ${txt.length>80?txt.substring(0,80)+'...':txt}';
      _messages.add({'role':'assistant','text':resp});
    } catch (e) {
      _messages.add({'role':'assistant','text':'Error: $e'});
    } finally {
      setState(()=>_loading=false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Visora Assistant'), backgroundColor: Colors.deepPurple),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.all(12),
              itemCount: _messages.length,
              itemBuilder: (_,i){
                final m = _messages[i];
                final isUser = m['role']=='user';
                return Row(
                  mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
                  children: [
                    Container(
                      margin: const EdgeInsets.symmetric(vertical:6),
                      padding: const EdgeInsets.all(12),
                      constraints: const BoxConstraints(maxWidth: 320),
                      decoration: BoxDecoration(color: isUser? Colors.deepPurple : Colors.white, borderRadius: BorderRadius.circular(12)),
                      child: Text(m['text']!, style: TextStyle(color: isUser? Colors.white : Colors.black87)),
                    )
                  ],
                );
              }
            ),
          ),
          Container(
            padding: const EdgeInsets.all(12),
            color: Colors.white,
            child: Row(
              children: [
                DropdownButton<String>(value: _tone, items: ['Motivational','Friendly','Formal','Funny'].map((t)=>DropdownMenuItem(value:t,child: Text(t))).toList(), onChanged: (v)=> setState(()=>_tone=v!)),
                const SizedBox(width:8),
                Expanded(child: TextField(controller: _ctrl, decoration: const InputDecoration(hintText:'Prompt likho...'))),
                IconButton(icon: _loading? const CircularProgressIndicator(): const Icon(Icons.send), onPressed: _loading?null:_sendPrompt),
              ],
            ),
          )
        ],
      ),
    );
  }
}
