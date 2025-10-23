import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'package:google_fonts/google_fonts.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});
  @override State<ProfileScreen> createState() => _ProfileScreenState();
}
class _ProfileScreenState extends State<ProfileScreen> {
  Map<String,dynamic>? _profile;
  bool _loading = true;

  @override void initState() { super.initState(); _load(); }
  Future<void> _load() async {
    try {
      final p = await ApiService.getProfile();
      setState((){ _profile = p; _loading=false; });
    } catch (e) { setState(()=>_loading=false); }
  }

  @override Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Profile'), backgroundColor: Colors.deepPurple),
      body: _loading ? const Center(child: CircularProgressIndicator()) :
      Padding(padding: const EdgeInsets.all(16), child: Column(
        children: [
          CircleAvatar(radius:40, backgroundImage: _profile!=null && _profile!['avatar']!=null ? NetworkImage(_profile!['avatar']) as ImageProvider : const AssetImage('assets/logo/visora_logo.png')),
          const SizedBox(height:12),
          Text(_profile?['name'] ?? 'User', style: GoogleFonts.poppins(fontSize:18,fontWeight: FontWeight.w700)),
          const SizedBox(height:6),
          Text(_profile?['email'] ?? '', style: const TextStyle(color: Colors.black54)),
          const SizedBox(height:12),
          Card(child: ListTile(title: const Text('Credits'), trailing: Text('${_profile?['credits'] ?? 0}'))),
          Card(child: ListTile(title: const Text('Plan'), trailing: Text('${_profile?['plan'] ?? 'Free'}'))),
          const SizedBox(height:12),
          ElevatedButton.icon(onPressed: () async { await ApiService.clearToken(); Navigator.pushNamedAndRemoveUntil(context, '/login', (r)=>false); }, icon: const Icon(Icons.logout), label: const Text('Logout'), style: ElevatedButton.styleFrom(backgroundColor: Colors.deepPurple)),
        ],
      )),
    );
  }
}
